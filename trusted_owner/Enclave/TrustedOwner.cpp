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

#include "TrustedOwner.h"
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include "Amd/amd_api.h"
#include "Amd/utilities.h"
#include "Amd/crypto.h"
#include "EnclaveAux.h"
#include "Enclave_t.h" // necessary for creating the binary
#include <sgx_utils.h>


// BLOB Structure is: 16 bytes (GUID) + 4 bytes (SIZE OF SECRETS TABLE) + 16 (SECRET_GUID) + 4 bytes (SIZE OF SECRET)
// + 1 (TERMINATOR '\0'). It sums to 57 bytes rounded up for a multiple of 16, which is 64 bytes --- the latter is a SEV-ES thing. 
#define SECRET_BLOB_LEN 64


void print_trusted_state() {
    unsigned char *p = (unsigned char *)&g_trusted_state;
    printf("Info: trusted state at %02x.\n", p);
        for (unsigned int i = 0; i < sizeof(trusted_owner_state_t); i++)
            printf("%02x ", p[i]);
    printf("\n");
}

void print_value(std::string msg, uint8_t* value, size_t size) {
    printf("Info: %s", msg);
    for (unsigned int i = 0; i < size; i++)
        printf("%02x ", value[i]);
    printf("\n");
}


int ecall_deploy_vm(sev_certs_t certs, measurement_info_t measurement_info, 
    sev_session_buf* out_session_data_buf, sev_cert* out_godh_pubkey_cert)
{
    if(g_trusted_state.status == init){
        // sev_session_buf session_data_buf;
        // sev_cert godh_pubkey_cert;
        printf("Info: session_data_buf = %02x.\n", out_session_data_buf);
        printf("Info: godh_pubkey_cert = %02x.\n", out_godh_pubkey_cert);

        print_trusted_state();

        printf("Info: starting ecall_deploy_vm ecall.\n");
        g_trusted_state.status = deployed;
        g_trusted_state.measurement_info = measurement_info;
        g_trusted_state.cek = certs.cek.pub_key.ecdsa;
        // MNONCE is not known at this stage so we just accept some
        // bogus value which is zeroed here
        memset(g_trusted_state.measurement_info.mnonce, 0, sizeof(nonce_128));
        printf("Info: validating certs.\n");
        int ret = amd_api_t::validate_cert_chain(&certs);
        if (ret != 0) {
            printf("Invalid certs.\n");
            return ret;
        }
        printf("Info: setting policy.\n");
        uint32_t policy = g_trusted_state.measurement_info.policy;
        print_value("Policy is : ", (uint8_t*) &policy, sizeof(policy));
	    printf("Info: setting pdh.\n");
        const sev_cert* pdh = &certs.pdh;
	    print_value("PDH is : ", (uint8_t*) pdh, sizeof(sev_cert));
        printf("Info: setting tks.\n");
        tek_tik* tks = &g_trusted_state.tks;
	    print_value("TKS is : ", (uint8_t*) tks, sizeof(tek_tik));
        printf("Info: calling generate_launch_blob.\n");
        ret = amd_api_t::generate_launch_blob(policy, pdh, out_session_data_buf, 
            out_godh_pubkey_cert, tks);
	if(ret != 0) {
		printf("Issue with blob generation");
	}
        print_trusted_state();
        printf("Info: finished ecall_deploy_vm ecall.\n");
    }
}

bool construct_secret_blob(uint8_t secret_blob[SECRET_BLOB_LEN], hmac_key_128* cik) {

    
    // Secrets GUID
    //unsigned char EFI_SECRET_TABLE_HEADER_GUID[16] = {0x1e, 0x74, 0xf5, 0x42, 0x71, 0xdd, 0x4d, 0x66, 0x96, 0x3e, 0xef, 0x42, 0x87, 0xff, 0x17, 0x3b};
    unsigned char EFI_SECRET_TABLE_HEADER_GUID[16] = {0x42, 0xf5, 0x74, 0x1e, 0xdd, 0x71, 0x66, 0x4d, 0x96, 0x3e, 0xef, 0x42, 0x87, 0xff, 0x17, 0x3b};
    // unsigned char CIK_GUID[16] = {0x73, 0x68, 0x69, 0xe5, 0x84, 0xf0, 0x49, 0x73, 0x92, 0xec, 0x06, 0x87, 0x9c, 0xe3, 0xda, 0x0b};
    unsigned char CIK_GUID[16] = {0xe5, 0x69, 0x68, 0x73, 0xf0, 0x84, 0x73, 0x49, 0x92, 0xec, 0x06, 0x87, 0x9c, 0xe3, 0xda, 0x0b};
    uint32_t secret_blob_len = SECRET_BLOB_LEN;
    uint32_t size_cik = 16 + 4 + 16;
    unsigned char TERMINATOR[1] = {'\0'};
    memcpy(secret_blob, EFI_SECRET_TABLE_HEADER_GUID, 16);
    memcpy(&secret_blob[16], &secret_blob_len, 4);
    memcpy(&secret_blob[20], CIK_GUID, 16);
    memcpy(&secret_blob[36], &size_cik, 4);
    memcpy(&secret_blob[40], cik, sizeof(hmac_key_128));

    print_value("secret_blob is : ", (uint8_t*) secret_blob, SECRET_BLOB_LEN);
}

int ecall_provision_vm(uint8_t hmac_measurement[32], nonce_128* mnonce,
    sev_hdr_buf* out_packaged_secret_header, uint8_t out_encrypted_blob[SECRET_BLOB_LEN], measurement_info_t* out_info)
{
    printf("Info: call to ecall_provision_vm ecall.\n");
    if(g_trusted_state.status == deployed){
        printf("Info: starting ecall_provision_vm ecall.\n");
        hmac_sha_256 calc_measurement {};
        measurement_t user_data {};
        user_data.meas_ctx = LAUNCH_MEASURE_CTX;
        user_data.api_major = g_trusted_state.measurement_info.api_major;
	    print_value("api major : ", (uint8_t*) &user_data.api_major, sizeof(user_data.api_major));
        user_data.api_minor = g_trusted_state.measurement_info.api_minor;
	    print_value("api minor : ", (uint8_t*) &user_data.api_minor, sizeof(user_data.api_minor));
        user_data.policy = g_trusted_state.measurement_info.policy;
	    print_value("policy : ", (uint8_t*) &user_data.policy, sizeof(user_data.policy));
	    user_data.build_id = g_trusted_state.measurement_info.build_id;
	    print_value("build_id : ", (uint8_t*) &user_data.build_id, sizeof(user_data.build_id));
        memcpy(user_data.digest, g_trusted_state.measurement_info.digest, 32);
	    print_value("digest : ", (uint8_t*) user_data.digest, sizeof(user_data.digest));
        memcpy(user_data.mnonce, mnonce, sizeof(nonce_128));
	    print_value("mnonce : ", (uint8_t*) user_data.mnonce, sizeof(user_data.mnonce));
        memcpy(g_trusted_state.measurement_info.mnonce, mnonce, sizeof(nonce_128));
        memcpy(user_data.tik, g_trusted_state.tks.tik, sizeof(aes_128_key));

        amd_api_t::calculate_measurement(&user_data, &calc_measurement);

        memset(&user_data, 0 , sizeof(measurement_t));

	    print_value("Measurement: ", (uint8_t*) calc_measurement, sizeof(hmac_sha_256));
       	print_value("hmac : ", hmac_measurement, sizeof(hmac_sha_256));	

        if(memcmp(calc_measurement,hmac_measurement, sizeof(hmac_sha_256)) == 0){
            printf("Info: measurement match.\n");
            tek_tik* tks = &g_trusted_state.tks;
            sev::gen_random_bytes(g_trusted_state.cik, sizeof(hmac_key_128));
            print_trusted_state();
            const uint8_t api_major = g_trusted_state.measurement_info.api_major; 
            const uint8_t api_minor = g_trusted_state.measurement_info.api_minor;

            uint8_t secret_blob[SECRET_BLOB_LEN] {};
            print_value("secret_blob is : ", (uint8_t*) secret_blob, SECRET_BLOB_LEN);
            construct_secret_blob(secret_blob, &g_trusted_state.cik);
            print_value("secret_blob is : ", (uint8_t*) secret_blob, SECRET_BLOB_LEN);

            amd_api_t::package_secret(tks, secret_blob, SECRET_BLOB_LEN, api_major, api_minor, 
                calc_measurement, out_packaged_secret_header, out_encrypted_blob);
            g_trusted_state.status = provisioned;
            memset(tks, 0 , sizeof(tek_tik_t));
            print_trusted_state();
            memcpy(out_info, &g_trusted_state.measurement_info, sizeof(measurement_info_t));
        }
    }
    printf("Info: end of ecall_provision_vm ecall.\n");
}

int ecall_generate_report_for_vm(uint8_t vm_data[64], hmac_sha_256* vm_data_hmac, sgx_target_info_t* qe_target_info,
    sgx_report_t* out_sgx_report)
{
    printf("Info: call to ecall_generate_report_for_vm ecall.\n");
    if(g_trusted_state.status == provisioned){
        printf("Info: starting ecall_generate_report_for_vm ecall.\n");
        printf("Info: ");
        hmac_sha_256 calc_data_hmac;
        #ifdef DEBUG
            printf("IN DEBUG\n");
            hmac_key_128 debug_cik;
            memset(debug_cik, 0 , sizeof(hmac_key_128));
            std::string debug_secret = "supersecret.";
            assert(debug_secret.size() <= sizeof(hmac_key_128));
            memcpy(debug_cik, debug_secret.c_str(), debug_secret.size());
            print_value("CIK is : ", debug_cik, sizeof(hmac_key_128));
            const hmac_key_128& cik = debug_cik;
        #else
             printf("NOT IN DEBUG\n");
            const hmac_key_128& cik = g_trusted_state.cik;
        #endif

        print_value("cik is : ", (uint8_t*) cik, sizeof(hmac_key_128));
        
        // Testing
        gen_hmac(&calc_data_hmac, cik, vm_data, 64);
        print_value("vm_data is : ", vm_data, 64);
        print_value("calc_data_hmac is : ", calc_data_hmac, sizeof(hmac_sha_256));
        print_value("vm_data_hmac is : ", *vm_data_hmac, sizeof(hmac_sha_256));

        // Verify HMAC
        if(memcmp(calc_data_hmac,*vm_data_hmac, sizeof(hmac_sha_256)) == 0){
            printf("Info: hmac match.\n");
            sgx_report_data_t sgx_report_data{};
            sgx_report_t sgx_report{};
            printf("Info: calc report data.\n");
            print_value("report data is : ", sgx_report_data.d, 64);
            calculate_report_data(vm_data, (uint8_t (*)[64])&sgx_report_data.d);
            print_value("report data is : ", sgx_report_data.d, 64);
            printf("Info: creating report.\n");
            sgx_create_report(qe_target_info, &sgx_report_data, out_sgx_report);
            printf("Info: Report created.\n");
        }
        printf("Info: finishing ecall_generate_report_for_vm ecall.\n");
    }
}

bool calculate_report_data(const uint8_t vm_data[64], uint8_t (*sgx_report_data)[64]) {
    bool error_raised = false;
    // Compute report data
    SHA256_CTX context;
    if (SHA256_Init(&context) != 1)
        return true;

    // cek
    const sev_ecdsa_pub_key* cek = &g_trusted_state.cek;
    if (SHA256_Update(&context, (void *)&cek->curve, sizeof(cek->curve)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&cek->qx, sizeof(cek->qx)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&cek->qy, sizeof(cek->qy)) != 1)
        return true;
    // The rmbz component of ecdsa is fixed at zero.

    // measurement_info
    const measurement_info_t* info = &g_trusted_state.measurement_info;
    if (SHA256_Update(&context, (void *)&info->api_major, sizeof(info->api_major)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info->api_minor, sizeof(info->api_minor)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info->build_id, sizeof(info->build_id)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info->policy, sizeof(info->policy)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info->digest, sizeof(info->digest)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info->mnonce, sizeof(info->mnonce)) != 1)
        return true;

    // vm_data
    if (SHA256_Update(&context, (void *)vm_data, 64) != 1)
        return true;

    if (SHA256_Final(*sgx_report_data, &context) != 1)
        return true;

    return error_raised;
}
    

