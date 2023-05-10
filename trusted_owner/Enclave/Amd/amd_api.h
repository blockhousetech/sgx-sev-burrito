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

#ifndef COMMANDS_H
#define COMMANDS_H

#include "sevapi.h"         // for hmac_sha_256, nonce_128, aes_128_key
#include "sevcore.h"        // for SEVDevice
#include <openssl/evp.h>    // for EVP_PKEY
#include <openssl/sha.h>    // for SHA256_DIGEST_LENGTH
#include <string>

const std::string PDH_FILENAME          = "pdh.cert";      // PDH signed by PEK
const std::string PDH_READABLE_FILENAME = "pdh_readable.txt";
const std::string PEK_FILENAME          = "pek.cert";      // PEK signed by CEK
const std::string PEK_READABLE_FILENAME = "pek_readable.txt";
const std::string OCA_FILENAME          = "oca.cert";      // OCA signed by P.O.
const std::string OCA_READABLE_FILENAME = "oca_readable.cert";
const std::string CEK_FILENAME          = "cek.cert";      // CEK signed by ASK
const std::string CEK_READABLE_FILENAME = "cek_readable.cert";
const std::string ASK_FILENAME          = "ask.cert";      // ASK signed by ARK
const std::string ASK_READABLE_FILENAME = "ask_readable.cert";
const std::string ARK_FILENAME          = "ark.cert";      // ARK self-signed
const std::string ARK_READABLE_FILENAME = "ark_readable.cert";

const std::string VCEK_DER_FILENAME               = "vcek.der";
const std::string VCEK_PEM_FILENAME               = "vcek.pem";
const std::string VCEK_CERT_CHAIN_PEM_FILENAME    = "cert_chain.pem";
const std::string VCEK_ASK_PEM_FILENAME           = "ask.pem";
const std::string VCEK_ARK_PEM_FILENAME           = "ark.pem";

const std::string CERTS_ZIP_FILENAME              = "certs_export";             // export_cert_chain
const std::string CERTS_VCEK_ZIP_FILENAME         = "certs_export_vcek";        // export_cert_chain_vcek
const std::string ASK_ARK_FILENAME                = "ask_ark.cert";             // get_ask_ark
const std::string PEK_CSR_HEX_FILENAME            = "pek_csr.cert";             // pek_csr
const std::string PEK_CSR_READABLE_FILENAME       = "pek_csr_readable.txt";     // pek_csr
const std::string SIGNED_PEK_CSR_FILENAME         = "pek_csr.signed.cert";      // sign_pek_csr
const std::string CERT_CHAIN_HEX_FILENAME         = "cert_chain.cert";          // pdh_cert_export
const std::string CERT_CHAIN_READABLE_FILENAME    = "cert_chain_readable.txt";  // pdh_cert_export
const std::string GET_ID_S0_FILENAME              = "getid_s0_out.txt";         // get_id
const std::string GET_ID_S1_FILENAME              = "getid_s1_out.txt";         // get_id
const std::string CALC_MEASUREMENT_READABLE_FILENAME = "calc_measurement_out.txt"; // calc_measurement
const std::string CALC_MEASUREMENT_FILENAME          = "calc_measurement_out.bin"; // calc_measurement
const std::string LAUNCH_BLOB_FILENAME            = "launch_blob.bin";          // generate_launch_blob
const std::string GUEST_OWNER_DH_FILENAME         = "godh.cert";                // generate_launch_blob
const std::string GUEST_TK_FILENAME               = "tmp_tk.bin";               // generate_launch_blob
const std::string SECRET_FILENAME                 = "secret.txt";               // package_secret
const std::string PACKAGED_SECRET_FILENAME        = "packaged_secret.bin";      // package_secret
const std::string PACKAGED_SECRET_HEADER_FILENAME = "packaged_secret_header.bin";// package_secret
const std::string ATTESTATION_REPORT_FILENAME     = "attestation_report.bin";   // validate_attestation
const std::string GUEST_REPORT_FILENAME           = "guest_report.bin";         // validate_guest_report

constexpr uint32_t BITS_PER_BYTE    = 8;
constexpr uint32_t NIST_KDF_H_BYTES = 32;
constexpr uint32_t NIST_KDF_H       = (NIST_KDF_H_BYTES*BITS_PER_BYTE); // 32*8=256
constexpr uint32_t NIST_KDF_R       = sizeof(uint32_t)*BITS_PER_BYTE;   // 32

constexpr uint8_t SEV_MASTER_SECRET_LABEL[] = "sev-master-secret";
constexpr uint8_t SEV_KEK_LABEL[]           = "sev-kek";
constexpr uint8_t SEV_KIK_LABEL[]           = "sev-kik";

constexpr auto LAUNCH_MEASURE_CTX           = 0x4;

struct measurement_t {
    uint8_t  meas_ctx;  // LAUNCH_MEASURE_CTX
    uint8_t  api_major;
    uint8_t  api_minor;
    uint8_t  build_id;
    uint32_t policy;    // SEV_POLICY
    uint8_t  digest[SHA256_DIGEST_LENGTH];   // gctx_ld
    nonce_128 mnonce;
    aes_128_key tik;
};

enum ccp_required_t {
    CCP_REQ     = 0,
    CCP_NOT_REQ = 1,
};

namespace amd_api_t {
    int calculate_measurement(const measurement_t* user_data, hmac_sha_256 *final_meas);
    int validate_cert_chain(sev_certs_t* certs);
    int generate_launch_blob(uint32_t policy, const sev_cert* pdh, sev_session_buf* out_session_data_buf, 
        sev_cert* out_godh_pubkey_cert, tek_tik* tks);
    int package_secret(const tek_tik* tks, const uint8_t* secret_blob, const size_t secret_blob_len, 
        const uint8_t api_major, const uint8_t api_minor, const hmac_sha_256 measurement, 
        sev_hdr_buf* out_packaged_secret_header, uint8_t* out_encrypted_secret_blob);
}

#endif /* COMMANDS_H */
