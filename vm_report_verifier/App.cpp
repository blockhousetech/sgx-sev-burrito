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


#include <stdio.h>
#include <vector>
#include <string.h>
#include <assert.h>
#include <fstream>
#include "sgx_urts.h"
#include "sgx_ql_quote.h"
#include "sgx_dcap_quoteverify.h"
#include "../trusted_owner/App/Amd.h"
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <limits>
#include <experimental/filesystem>
#include <iostream>


using namespace std;

void print_struct_memory(unsigned char *p, size_t size) {
    printf("Info: printing struct at %02x.\n", p);
        for (unsigned int i = 0; i < size; i++)
            printf("%02x ", p[i]);
    printf("\n");
}

void print_value(std::string msg, uint8_t* value, size_t size) {
    printf("%s ", msg.data());
    for (unsigned int i = 0; i < size; i++)
        printf("%02x", value[i]);
    printf("\n");
}

vector<uint8_t> readBinaryContent(const string& filePath)
{
    ifstream file(filePath, ios::binary);
    if (!file.is_open())
    {
        printf("Error: Unable to open quote file %s\n", filePath.c_str());
        return {};
    }

    file.seekg(0, ios_base::end);
    streampos fileSize = file.tellg();
    file.seekg(0, ios_base::beg);
    vector<uint8_t> retVal(fileSize);
    file.read(reinterpret_cast<char*>(retVal.data()), fileSize);
    file.close();
    return retVal;
}
#define PATHSIZE 0x100


/**
 * @param quote - ECDSA quote buffer
 * @param use_qve - Set quote verification mode
 *                   If true, quote verification will be performed by Intel QvE
 *                   If false, quote verification will be performed by untrusted QVL
 */

int ecdsa_quote_verification(vector<uint8_t> quote)
{
    int ret = 0;
    time_t current_time = 0;
    uint32_t supplemental_data_size = 0;
    uint8_t *p_supplemental_data = NULL;
    sgx_status_t sgx_ret = SGX_SUCCESS;
    quote3_error_t dcap_ret = SGX_QL_ERROR_UNEXPECTED;
    sgx_ql_qv_result_t quote_verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;
    uint32_t collateral_expiration_status = 1;

    // Untrusted quote verification
 
    //call DCAP quote verify library to get supplemental data size
    //
    dcap_ret = sgx_qv_get_quote_supplemental_data_size(&supplemental_data_size);
    if (dcap_ret == SGX_QL_SUCCESS && supplemental_data_size == sizeof(sgx_ql_qv_supplemental_t)) {
        printf("\tInfo: sgx_qv_get_quote_supplemental_data_size successfully returned.\n");
        p_supplemental_data = (uint8_t*)malloc(supplemental_data_size);
        if (p_supplemental_data != NULL) {
            memset(p_supplemental_data, 0, sizeof(supplemental_data_size));
        }
        //Just print error in sample
        //
        else {
            printf("\tError: Cannot allocate memory for supplemental data.\n");
        }
    }
    else {
        if (dcap_ret != SGX_QL_SUCCESS)
            printf("\tError: sgx_qv_get_quote_supplemental_data_size failed: 0x%04x\n", dcap_ret);

        if (supplemental_data_size != sizeof(sgx_ql_qv_supplemental_t))
            printf("\tWarning: Quote supplemental data size is different between DCAP QVL and QvE, please make sure you installed DCAP QVL and QvE from same release.\n");

        supplemental_data_size = 0;
    }

    //set current time. This is only for sample purposes, in production mode a trusted time should be used.
    //
    current_time = time(NULL);

    // print_struct_memory((unsigned char *)quote.data(), quote.size());

    //call DCAP quote verify library for quote verification
    //here you can choose 'trusted' or 'untrusted' quote verification by specifying parameter '&qve_report_info'
    //if '&qve_report_info' is NOT NULL, this API will call Intel QvE to verify quote
    //if '&qve_report_info' is NULL, this API will call 'untrusted quote verify lib' to verify quote, this mode doesn't rely on SGX capable system, but the results can not be cryptographically authenticated
    dcap_ret = sgx_qv_verify_quote(
        quote.data(), (uint32_t)quote.size(),
        NULL,
        current_time,
        &collateral_expiration_status,
        &quote_verification_result,
        NULL,
        supplemental_data_size,
        p_supplemental_data);
    if (dcap_ret == SGX_QL_SUCCESS) {
        printf("\tInfo: App: sgx_qv_verify_quote successfully returned.\n");
    }
    else {
        printf("\tError: App: sgx_qv_verify_quote failed: 0x%04x\n", dcap_ret);
    }

    //check verification result
    //
    switch (quote_verification_result)
    {
    case SGX_QL_QV_RESULT_OK:
        //check verification collateral expiration status
        //this value should be considered in your own attestation/verification policy
        //
        if (collateral_expiration_status == 0) {
            printf("\tInfo: App: Verification completed successfully.\n");
            ret = 0;
        }
        else {
            printf("\tWarning: App: Verification completed, but collateral is out of date based on 'expiration_check_date' you provided.\n");
            ret = 1;
        }
        break;
    case SGX_QL_QV_RESULT_CONFIG_NEEDED:
    case SGX_QL_QV_RESULT_OUT_OF_DATE:
    case SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED:
    case SGX_QL_QV_RESULT_SW_HARDENING_NEEDED:
    case SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED:
        printf("\tWarning: App: Verification completed with Non-terminal result: %x\n", quote_verification_result);
        ret = 1;
        break;
    case SGX_QL_QV_RESULT_INVALID_SIGNATURE:
    case SGX_QL_QV_RESULT_REVOKED:
    case SGX_QL_QV_RESULT_UNSPECIFIED:
    default:
        printf("\tError: App: Verification completed with Terminal result: %x\n", quote_verification_result);
        ret = -1;
        break;
    }

    //check supplemental data if necessary
    //
    if (p_supplemental_data != NULL && supplemental_data_size > 0) {
        sgx_ql_qv_supplemental_t *p = (sgx_ql_qv_supplemental_t*)p_supplemental_data;

        //you can check supplemental data based on your own attestation/verification policy
        //here we only print supplemental data version for demo usage
        //
        printf("\tInfo: Supplemental data version: %d\n", p->version);
    }

    if (p_supplemental_data) {
        free(p_supplemental_data);
    }

    return ret;
}

bool calculate_report_data(const uint8_t vm_data[64], const sev_ecdsa_pub_key& cek, const measurement_info_t& info, uint8_t out_report_data[64]) {
    bool error_raised = false;

    // Compute report data
    SHA256_CTX context;
    if (SHA256_Init(&context) != 1)
        return true;

    // cek
    if (SHA256_Update(&context, (void *)&cek.curve, sizeof(cek.curve)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&cek.qx, sizeof(cek.qx)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&cek.qy, sizeof(cek.qy)) != 1)
        return true;
    
    // The rmbz component of ecdsa is fixed at zero.
    printf("\tInformation used in the calculation of measurement:\n");
    print_value("\t  cek curve ", (uint8_t*)&cek.curve, sizeof(cek.curve));
    print_value("\t  cek qx ", (uint8_t*)&cek.qx, sizeof(cek.qx));
    print_value("\t  cek qy ", (uint8_t*)&cek.qy, sizeof(cek.qy));

    // measurement_info
    if (SHA256_Update(&context, (void *)&info.api_major, sizeof(info.api_major)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info.api_minor, sizeof(info.api_minor)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info.build_id, sizeof(info.build_id)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info.policy, sizeof(info.policy)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info.digest, sizeof(info.digest)) != 1)
        return true;
    if (SHA256_Update(&context, (void *)&info.mnonce, sizeof(info.mnonce)) != 1)
        return true;

    print_value("\t  api major ", (uint8_t*)&info.api_major, sizeof(info.api_major));
    print_value("\t  api minor ", (uint8_t*)&info.api_minor, sizeof(info.api_minor));
    print_value("\t  build id  ", (uint8_t*)&info.build_id, sizeof(info.build_id));
    print_value("\t  policy ", (uint8_t*)&info.policy, sizeof(info.policy));
    print_value("\t  digest ", (uint8_t*)&info.digest, sizeof(info.digest));
    print_value("\t  nonce ", (uint8_t*)&info.mnonce, sizeof(info.mnonce));

    print_value("\t  vm data ", (uint8_t*)vm_data, 64);

    // vm_data
    if (SHA256_Update(&context, (void *)vm_data, 64*sizeof(uint8_t)) != 1)
        return true;

    if (SHA256_Final(out_report_data, &context) != 1)
        return true;

    return error_raised;
}

bool verify_report(const vector<uint8_t>& raw_quote, const uint8_t mr_enclave[32], const uint8_t vm_data[64], const sev_cert& cek_cert, const measurement_info_t& info){

    const sgx_quote3_t& quote = reinterpret_cast<const sgx_quote3_t&>(*raw_quote.data());
    uint8_t cal_report_data[64] {};
    const sev_ecdsa_pub_key& cek = cek_cert.pub_key.ecdsa;
    calculate_report_data(vm_data, cek, info, cal_report_data);

    if(memcmp(cal_report_data, quote.report_body.report_data.d, 64) == 0){
        printf("Maching vm data \n");
    } else {
        printf("Report data does not match.\n");
        return false;
    }

    if(memcmp(mr_enclave, quote.report_body.mr_enclave.m, 32) == 0){
        printf("Maching mrenclave \n");
    } else {
        printf("Report mrenclave does not match.\n");
        return false;
    }
    
    return true;
}


void usage()
{
    printf("\nUsage:\n");
    printf("\t./verifier $MR_ENCLAVE$ $VM_DATA_IN_BASE64$ $PATH_TO_FOLDER_WITH_CEK_AND_QUOTE$ \n");
}

// Base64 code obtained from: https://github.com/elzoughby/Base64

char base64_map[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                     'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                     'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

char* base64_decode(char* cipher) {

    char counts = 0;
    char buffer[4];
    char* plain = (char*) malloc(strlen(cipher) * 3 / 4);
    int i = 0, p = 0;

    for(i = 0; cipher[i] != '\0'; i++) {
        char k;
        for(k = 0 ; k < 64 && base64_map[k] != cipher[i]; k++);
        buffer[counts++] = k;
        if(counts == 4) {
            plain[p++] = (buffer[0] << 2) + (buffer[1] >> 4);
            if(buffer[2] != 64)
                plain[p++] = (buffer[1] << 4) + (buffer[2] >> 2);
            if(buffer[3] != 64)
                plain[p++] = (buffer[2] << 6) + buffer[3];
            counts = 0;
        }
    }

    return plain;
}


/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    if (argc != 4) {
        usage();
        return 1;
    }

    printf("Mrenclave is %s\n", argv[1]);
    char* mr_enclave_dec = base64_decode(argv[1]);
    uint8_t mr_enclave[32]{};
    memcpy(mr_enclave, mr_enclave_dec, 32*sizeof(uint8_t));
    print_value("mr_enclave_dec=", (uint8_t*) mr_enclave, 32);

    printf("VM data is %s\n", argv[2]);
    char* vm_data_dec = base64_decode(argv[2]);
    uint8_t vm_data[64]{};
    memcpy(vm_data, vm_data_dec, 64*sizeof(uint8_t));
    print_value("value dec=", (uint8_t*) vm_data, 64);

    const size_t MAX_PATH_SIZE = 1000;
    size_t path_len = strnlen(argv[3], MAX_PATH_SIZE);
    string folder_path(argv[3],path_len);
    
    if (folder_path.empty()) {
         usage();
    }

    string quote_path = folder_path + "quote.dat";
    //read quote from file
    vector<uint8_t> quote = readBinaryContent(quote_path);
    if (quote.empty()) {
        usage();
        return 1;
    }

    printf("\nQuote verification without QvE:\n");

    if(ecdsa_quote_verification(quote) == -1){
        printf("Quote authentication failed.\n");
        return 1;
    }

    string cek_cert_path = folder_path + "cek.cert";
    uint8_t cek_cert_raw[sizeof(sev_cert)] {};
    amd_api_u::read_file(cek_cert_path, cek_cert_raw, sizeof(sev_cert));
    const sev_cert& cek = reinterpret_cast<const sev_cert&>(cek_cert_raw);

    string measurement_info_path = folder_path + "info.dat";
    uint8_t measurement_info_raw[sizeof(measurement_info_t)] {};
    amd_api_u::read_file(measurement_info_path, measurement_info_raw, sizeof(measurement_info_t));
    const measurement_info_t& info = reinterpret_cast<const measurement_info_t&>(measurement_info_raw);

    if(!verify_report(quote, mr_enclave, vm_data, cek, info)){
        printf("\nQuote report data does not match.\n");
        return 1;
    }

    printf("\nQuote verified.\n");
    printf("\n");

    return 0;
}
