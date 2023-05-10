/*
 * Burrito
 * Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "amdcert.h"
#include "amd_api.h"
#include "crypto.h"
#include "sevcert.h"
#include "../EnclaveAux.h"
#include "utilities.h"      // for WriteToFile
#include <openssl/hmac.h>   // for calc_measurement
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <stdlib.h>         // malloc


const bool VERBOSE = false;



namespace {

    // --------------------------------------------------------------- //
    // ---------------- generate_launch_blob functions --------------- //
    // --------------------------------------------------------------- //
    /*
    * NIST Compliant KDF
    */
    bool kdf(uint8_t *key_out,       size_t key_out_length,
                    const uint8_t *key_in,  size_t key_in_length,
                    const uint8_t *label,   size_t label_length,
                    const uint8_t *context, size_t context_length)
    {
        if (!key_out || !key_in || !label || (key_out_length == 0) ||
        (key_in_length == 0) || (label_length == 0))
            return false;

        bool cmd_ret = false;
        uint8_t null_byte = '\0';
        unsigned int out_len = 0;
        uint8_t prf_out[NIST_KDF_H_BYTES];      // Buffer to collect PRF output

        // length in bits of derived key
        uint32_t l = (uint32_t)(key_out_length * BITS_PER_BYTE);

        // number of iterations to produce enough derived key bits
        uint32_t n = ((l - 1) / NIST_KDF_H) + 1;

        size_t bytes_left = key_out_length;
        uint32_t offset = 0;

        // Create and initialize the context
        HMAC_CTX *ctx;
        if (!(ctx = HMAC_CTX_new()))
            return cmd_ret;

        for (unsigned int i = 1; i <= n; i++)
        {
            if (HMAC_CTX_reset(ctx) != 1)
                break;

            // calculate a chunk of random data from the PRF
            if (HMAC_Init_ex(ctx, key_in, (int)key_in_length, EVP_sha256(), NULL) != 1)
                break;
            if (HMAC_Update(ctx, (uint8_t *)&i, sizeof(i)) != 1)
                break;
            if (HMAC_Update(ctx, (unsigned char*)label, label_length) != 1)
                break;
            if (HMAC_Update(ctx, &null_byte, sizeof(null_byte)) != 1)
                break;
            if ((context) && (context_length != 0)) {
                if (HMAC_Update(ctx, (unsigned char*)context, context_length) != 1)
                    break;
            }
            if (HMAC_Update(ctx, (uint8_t *)&l, sizeof(l)) != 1)
                break;
            if (HMAC_Final(ctx, prf_out, &out_len) != 1)
                break;

            // Write out the key bytes
            if (bytes_left <= NIST_KDF_H_BYTES) {
                memcpy(key_out + offset, prf_out, bytes_left);
            }
            else {
                memcpy(key_out + offset, prf_out, NIST_KDF_H_BYTES);
                offset     += NIST_KDF_H_BYTES;
                bytes_left -= NIST_KDF_H_BYTES;
            }

            if (i == n)          // If successfully finished all iterations
                cmd_ret = true;
        }

        HMAC_CTX_free(ctx);
        return cmd_ret;
    }

    /*
    * Note that this function NEWs/allocates memory for a
    * uint8_t array using OPENSSL_malloc that must be freed
    * in the calling function using OPENSSL_FREE()
    */
    uint8_t * calculate_shared_secret(EVP_PKEY *priv_key, EVP_PKEY *peer_key,
                                            size_t& shared_key_len_out)
    {
        if (!priv_key || !peer_key)
            return NULL;

        bool success = false;
        EVP_PKEY_CTX *ctx = NULL;
        uint8_t *shared_key = NULL;

        do {
            // Create the context using your private key
            if (!(ctx = EVP_PKEY_CTX_new(priv_key, NULL)))
                break;

            // Calculate the intermediate secret
            if (EVP_PKEY_derive_init(ctx) <= 0)
                break;
            if (EVP_PKEY_derive_set_peer(ctx, peer_key) <= 0)
                break;

            // Determine buffer length
            if (EVP_PKEY_derive(ctx, NULL, &shared_key_len_out) <= 0)
                break;

            // Need to free shared_key using OPENSSL_FREE() in the calling function
            shared_key = (unsigned char*)OPENSSL_malloc(shared_key_len_out);
            if (!shared_key)
                break;      // malloc failure

            // Compute the shared secret with the ECDH key material.
            if (EVP_PKEY_derive(ctx, shared_key, &shared_key_len_out) <= 0)
                break;

            success = true;
        } while (0);

        EVP_PKEY_CTX_free(ctx);

        return success ? shared_key : NULL;
    }

    /*
    * Generate a master_secret value from our (test suite) Private DH key,
    *   the Platform's public DH key, and a nonce
    * This function calls two functions (above) which allocate memory
    *   for keys, and this function must free that memory
    */
    bool derive_master_secret(aes_128_key master_secret,
                                    EVP_PKEY *godh_priv_key,
                                    const sev_cert *pdh_public,
                                    const uint8_t nonce[sizeof(nonce_128)])
    {
        if (!godh_priv_key || !pdh_public)
            return false;

        sev_cert dummy;
        memset(&dummy, 0, sizeof(sev_cert));    // To remove compile warnings
        SEVCert temp_obj(&dummy);                // TODO. Hack b/c just want to call function later
        bool ret = false;
        EVP_PKEY *plat_pub_key = NULL;    // Platform public key
        size_t shared_key_len = 0;

        do {
            // New up the Platform's public EVP_PKEY
            if (!(plat_pub_key = EVP_PKEY_new()))
                break;

            // Get the friend's Public EVP_PKEY from the certificate
            // This function allocates memory and attaches an EC_Key
            //  to your EVP_PKEY so, to prevent mem leaks, make sure
            //  the EVP_PKEY is freed at the end of this function
            if (temp_obj.compile_public_key_from_certificate(pdh_public, plat_pub_key) != STATUS_SUCCESS)
                break;

            // Calculate the shared secret
            // This function is allocating memory for this uint8_t[],
            //  must free it at the end of this function
            uint8_t *shared_key = calculate_shared_secret(godh_priv_key, plat_pub_key, shared_key_len);
            if (!shared_key)
                break;

            // Derive the master secret from the intermediate secret
            if (!kdf((unsigned char*)master_secret, sizeof(aes_128_key), shared_key,
                shared_key_len, (uint8_t *)SEV_MASTER_SECRET_LABEL,
                sizeof(SEV_MASTER_SECRET_LABEL)-1, nonce, sizeof(nonce_128))) // sizeof(nonce), bad?
                break;

            // Free memory allocated in calculate_shared_secret
            OPENSSL_free(shared_key);    // Local variable

            ret = true;
        } while (0);

        // Free memory
        EVP_PKEY_free(plat_pub_key);

        return ret;
    }

    bool derive_kek(aes_128_key kek, const aes_128_key master_secret)
    {
        bool ret = kdf((unsigned char*)kek, sizeof(aes_128_key), master_secret, sizeof(aes_128_key),
                    (uint8_t *)SEV_KEK_LABEL, sizeof(SEV_KEK_LABEL)-1, NULL, 0);
        return ret;
    }

    bool derive_kik(hmac_key_128 kik, const aes_128_key master_secret)
    {
        bool ret = kdf((unsigned char*)kik, sizeof(aes_128_key), master_secret, sizeof(aes_128_key),
                    (uint8_t *)SEV_KIK_LABEL, sizeof(SEV_KIK_LABEL)-1, NULL, 0);
        return ret;
    }

    bool gen_hmac(hmac_sha_256 *out, hmac_key_128 key, uint8_t *msg, size_t msg_len)
    {
        if (!out || !msg)
            return false;

        unsigned int out_len = 0;
        HMAC(EVP_sha256(), key, sizeof(hmac_key_128), msg,    // Returns NULL or value of out
            msg_len, (uint8_t *)out, &out_len);

        if ((out != NULL) && (out_len == sizeof(hmac_sha_256)))
            return true;
        else
            return false;
    }

    /*
    * AES128 Encrypt a buffer
    */
    bool encrypt(uint8_t *out, const uint8_t *in, size_t length,
                        const aes_128_key Key, const iv_128 IV)
    {
        if (!out || !in)
            return false;

        EVP_CIPHER_CTX *ctx;
        int len = 0;
        bool cmd_ret = false;

        do {
            // Create and initialize the context
            if (!(ctx = EVP_CIPHER_CTX_new()))
                break;

            // Initialize the encryption operation. IMPORTANT - ensure you
            // use a key and IV size appropriate for your cipher
            if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), NULL, Key, IV) != 1)
                break;

            // Provide the message to be encrypted, and obtain the encrypted output
            if (EVP_EncryptUpdate(ctx, out, &len, in, (int)length) != 1)
                break;

            // Finalize the encryption. Further out bytes may be written at
            // this stage
            if (EVP_EncryptFinal_ex(ctx, out + len, &len) != 1)
                break;

            cmd_ret = true;
        } while (0);

        // Clean up
        EVP_CIPHER_CTX_free(ctx);

        return cmd_ret;
    }

    int build_session_buffer(sev_session_buf *buf, uint32_t guest_policy,
                                    EVP_PKEY *godh_priv_key, const sev_cert *pdh_pub, tek_tik* tks)
    {
        int cmd_ret = -1;

        aes_128_key master_secret;
        nonce_128 nonce;
        aes_128_key kek;
        hmac_key_128 kik;
        iv_128 iv;
        tek_tik wrap_tk;
        hmac_sha_256 wrap_mac;
        hmac_sha_256 policy_mac;

        do {
            // Generate a random nonce
            sev::gen_random_bytes(nonce, sizeof(nonce_128));

            // Derive Master Secret
            if (!derive_master_secret(master_secret, godh_priv_key, pdh_pub, nonce))
                break;

            // Derive the KEK and KIK
            if (!derive_kek(kek, master_secret))
                break;
            if (!derive_kik(kik, master_secret))
                break;

            // Generate a random TEK and TIK. Combine in to TK. Wrap.
            // Preserve TK for use in LAUNCH_MEASURE and LAUNCH_SECRET
            sev::gen_random_bytes(tks->tek, sizeof(tks->tek));
            sev::gen_random_bytes(tks->tik, sizeof(tks->tik));

            // Create an IV and wrap the TK with KEK and IV
            sev::gen_random_bytes(iv, sizeof(iv_128));
            if (!encrypt((uint8_t *)&wrap_tk, (uint8_t *)tks, sizeof(tek_tik), kek, iv))
                break;

            // Generate the HMAC for the wrap_tk
            if (!gen_hmac(&wrap_mac, kik, (uint8_t *)&wrap_tk, sizeof(wrap_tk)))
                break;

            // Generate the HMAC for the Policy bits
            if (!gen_hmac(&policy_mac, tks->tik, (uint8_t *)&guest_policy, sizeof(guest_policy)))
                break;

            // Copy everything to the session data buffer
            memcpy(&buf->nonce, &nonce, sizeof(buf->nonce));
            memcpy(&buf->wrap_tk, &wrap_tk, sizeof(buf->wrap_tk));
            memcpy(&buf->wrap_iv, &iv, sizeof(buf->wrap_iv));
            memcpy(&buf->wrap_mac, &wrap_mac, sizeof(buf->wrap_mac));
            memcpy(&buf->policy_mac, &policy_mac, sizeof(buf->policy_mac));

            cmd_ret = STATUS_SUCCESS;
        } while (0);

        return cmd_ret;
    }

    // --------------------------------------------------------------- //
    // ------------------- package_secret functions ------------------ //
    // --------------------------------------------------------------- //

    bool create_launch_secret_header(sev_hdr_buf *out_header, iv_128 *iv,
                                            uint8_t *buf, size_t buffer_len,
                                            uint32_t hdr_flags,
                                            uint8_t api_major, uint8_t api_minor, const hmac_sha_256 measurement, const tek_tik* tks)
    {
        bool ret = false;

        // Note: API <= 0.16 and older does LaunchSecret differently than Naples API >= 0.17
        const uint8_t meas_ctx = 0x01;
        sev_hdr_buf header;
        uint32_t measurement_length = sizeof(header.mac);
        const uint32_t buf_len = (uint32_t)buffer_len;
	    printf("Buffer len is: %d \n", buf_len);
        memcpy(header.iv, iv, sizeof(iv_128));
        header.flags = hdr_flags;

        // Create and initialize the context
        HMAC_CTX *ctx;
        if (!(ctx = HMAC_CTX_new()))
            return ret;

        do {
            if (HMAC_Init_ex(ctx, tks->tik, sizeof(tks->tik), EVP_sha256(), NULL) != 1)
                break;
            if (HMAC_Update(ctx, &meas_ctx, sizeof(meas_ctx)) != 1)
                break;
            if (HMAC_Update(ctx, (uint8_t *)&header.flags, sizeof(header.flags)) != 1)
                break;
            if (HMAC_Update(ctx, (uint8_t *)&header.iv, sizeof(header.iv)) != 1)
                break;
            if (HMAC_Update(ctx, (uint8_t *)&buf_len, sizeof(buf_len)) != 1) // Guest Length
                break;
            if (HMAC_Update(ctx, (uint8_t *)&buf_len, sizeof(buf_len)) != 1) // Trans Length
                break;
            if (HMAC_Update(ctx, buf, buf_len) != 1)                        // Data
                break;
            if (sev::min_api_version(api_major, api_minor, 0, 17)) {
		       printf("Size of measurement is: %d \n",sizeof(hmac_sha_256));
                if (HMAC_Update(ctx, measurement, sizeof(hmac_sha_256)) != 1) // Measure
                    break;
            }
            if (HMAC_Final(ctx, (uint8_t *)&header.mac, &measurement_length) != 1)
                break;

            memcpy(out_header, &header, sizeof(sev_hdr_buf));
	        printf("Header is right!");
            ret = true;
        } while (0);

        HMAC_CTX_free(ctx);

        return ret;
    }
}

namespace amd_api_t {

    // We cannot call LaunchMeasure to get the MNonce because that command doesn't
    // exist in this context, so we read the user input params for all of our data
    int calculate_measurement(const measurement_t *user_data, hmac_sha_256 *final_meas)
    {
        int cmd_ret = ERROR_UNSUPPORTED;

        uint32_t measurement_length = sizeof(final_meas);

        // Create and initialize the context
        HMAC_CTX *ctx;
        if (!(ctx = HMAC_CTX_new()))
            return ERROR_BAD_MEASUREMENT;

        do {
            if (HMAC_Init_ex(ctx, user_data->tik, sizeof(user_data->tik), EVP_sha256(), NULL) != 1)
                break;
            if (sev::min_api_version(user_data->api_major, user_data->api_minor, 0, 17)) {
                if (HMAC_Update(ctx, &user_data->meas_ctx, sizeof(user_data->meas_ctx)) != 1)
                    break;
                if (HMAC_Update(ctx, &user_data->api_major, sizeof(user_data->api_major)) != 1)
                    break;
                if (HMAC_Update(ctx, &user_data->api_minor, sizeof(user_data->api_minor)) != 1)
                    break;
                if (HMAC_Update(ctx, &user_data->build_id, sizeof(user_data->build_id)) != 1)
                    break;
            }
            if (HMAC_Update(ctx, (uint8_t *)&user_data->policy, sizeof(user_data->policy)) != 1)
                break;
            if (HMAC_Update(ctx, (uint8_t *)&user_data->digest, sizeof(user_data->digest)) != 1)
                break;
            // Use the same random MNonce as the FW in our validation calculations
            if (HMAC_Update(ctx, (uint8_t *)&user_data->mnonce, sizeof(user_data->mnonce)) != 1)
                break;
            if (HMAC_Final(ctx, (uint8_t *)final_meas, &measurement_length) != 1)  // size = 32
                break;

            cmd_ret = STATUS_SUCCESS;
        } while (0);

        HMAC_CTX_free(ctx);
        return cmd_ret;
    }

    int validate_cert_chain(sev_certs_t* certs)
    {
        int cmd_ret = -1;

        sev_cert ask_pubkey;

        do {

            // Temp structs because they are class functions
            SEVCert tmp_sev_cek(&certs->cek);   // Pass in child cert in constructor
            SEVCert tmp_sev_pek(&certs->pek);
            SEVCert tmp_sev_pdh(&certs->pdh);
            AMDCert tmp_amd;

            // Validate the ARK
            cmd_ret = tmp_amd.amd_cert_validate_ark(&certs->ark);
            if (cmd_ret != STATUS_SUCCESS)
                break;

            // Validate the ASK
            cmd_ret = tmp_amd.amd_cert_validate_ask(&certs->ask, &certs->ark);
            if (cmd_ret != STATUS_SUCCESS)
                break;

            // Export the ASK to an AMD cert public key
            // The verify_sev_cert function takes in a parent of an sev_cert not
            //   an amd_cert, so need to pull the pubkey out of the amd_cert and
            //   place it into a tmp sev_cert to help validate the cek
            cmd_ret = tmp_amd.amd_cert_export_pub_key(&certs->ask, &ask_pubkey);
            if (cmd_ret != STATUS_SUCCESS)
                break;

            // Validate the CEK
            cmd_ret = tmp_sev_cek.verify_sev_cert(&ask_pubkey);
            if (cmd_ret != STATUS_SUCCESS)
                break;

            // Validate the PEK with the CEK and OCA
            cmd_ret = tmp_sev_pek.verify_sev_cert(&certs->cek, &certs->oca);
            if (cmd_ret != STATUS_SUCCESS)
                break;

            // Validate the PDH
            cmd_ret = tmp_sev_pdh.verify_sev_cert(&certs->pek);
            if (cmd_ret != STATUS_SUCCESS)
                break;
        } while (0);

        return (int)cmd_ret;
    }

    int generate_launch_blob(uint32_t policy, const sev_cert* pdh, sev_session_buf* out_session_data_buf, 
        sev_cert* out_godh_pubkey_cert, tek_tik* tks)
    {
        printf("generate_launch_blob \n");
        int cmd_ret = ERROR_UNSUPPORTED;
        EVP_PKEY *godh_key_pair = NULL;      // Guest Owner Diffie-Hellman

        printf("Zeroing outputs \n");

        printf("Starting processing %02x \n", out_godh_pubkey_cert);

        memset(out_session_data_buf, 0, sizeof(sev_session_buf));
        memset(out_godh_pubkey_cert, 0, sizeof(sev_cert));
        memset(tks, 0, sizeof(tek_tik));

        do {
            printf("Starting processing %02x \n", out_godh_pubkey_cert);

            // Launch Start needs the GODH Pubkey as a cert, so need to create it
            SEVCert cert_obj(out_godh_pubkey_cert);

            printf("Keypair before: %02x \n", godh_key_pair);

            printf("Calling generate_ecdh_key_pair \n");
            // Generate a new GODH Public/Private keypair
            if (!generate_ecdh_key_pair(&godh_key_pair)) {
                printf("Error generating new GODH ECDH keypair\n");
                break;
            }

            printf("Calling generate_ecdh_key_pair \n");

            printf("Keypair: %02x \n", godh_key_pair);
            printf("Cert obj: %02x \n", cert_obj);
            // This cert is really just a way to send over the godh public key,
            // so the api major/minor don't matter here
            if (!cert_obj.create_godh_cert(&godh_key_pair, 0, 0)) {
                printf("Error creating GODH certificate\n");
                break;
            }
            printf("memcpying \n");
            memcpy(out_godh_pubkey_cert, cert_obj.data(), sizeof(sev_cert)); // TODO, shouldn't need this?

            printf("Calling build_session_buffer \n");
            
            cmd_ret = build_session_buffer(out_session_data_buf, policy, godh_key_pair, pdh, tks);
            if (cmd_ret == STATUS_SUCCESS) {
                    printf("Guest Policy (input): %08x\n", policy);
                    printf("nonce:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->nonce); i++) {
                        printf("%02x ", out_session_data_buf->nonce[i]);
                    }
                    printf("\nWrapTK TEK:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->wrap_tk.tek); i++) {
                        printf("%02x ", out_session_data_buf->wrap_tk.tek[i]);
                    }
                    printf("\nWrapTK TIK:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->wrap_tk.tik); i++) {
                        printf("%02x ", out_session_data_buf->wrap_tk.tik[i]);
                    }
                    printf("\nWrapIV:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->wrap_iv); i++) {
                        printf("%02x ", out_session_data_buf->wrap_iv[i]);
                    }
                    printf("\nWrapMAC:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->wrap_mac); i++) {
                        printf("%02x ", out_session_data_buf->wrap_mac[i]);
                    }
                    printf("\nPolicyMAC:\n");
                    for (size_t i = 0; i < sizeof(out_session_data_buf->policy_mac); i++) {
                        printf("%02x ", out_session_data_buf->policy_mac[i]);
                    }
                    printf("\n");
                }

                printf("Finished lauch blob \n");

        } while (0);

        return (int)cmd_ret;
    }

    int package_secret(const tek_tik* tks, const uint8_t* secret_blob, const size_t secret_blob_len, 
        const uint8_t api_major, const uint8_t api_minor, const hmac_sha_256 measurement, 
        sev_hdr_buf* out_packaged_secret_header, uint8_t* out_encrypted_secret_blob)
    {
        int cmd_ret = ERROR_UNSUPPORTED;
        sev_hdr_buf packaged_secret_header;

        uint32_t flags = 0;
        iv_128 iv;
        sev::gen_random_bytes(&iv, sizeof(iv));     // Pick a random IV

        do {
            // Encrypt the secret with the TEK
            encrypt((uint8_t*)out_encrypted_secret_blob, secret_blob, secret_blob_len, tks->tek, iv);

            // Set up the Launch_Secret packet header
            if (!create_launch_secret_header(out_packaged_secret_header, &iv, out_encrypted_secret_blob,
                                            secret_blob_len, flags,
                                            api_major, api_minor, measurement, tks)) {
                break;
            }

            cmd_ret = STATUS_SUCCESS;
        } while (0);

        return (int)cmd_ret;
    }
}

