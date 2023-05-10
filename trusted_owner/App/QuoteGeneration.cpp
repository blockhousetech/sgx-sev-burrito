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

/**
 * File: app.cpp
 *
 * Description: Sample application to
 * demonstrate the usage of quote generation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "QuoteGeneration.h"
#include "sgx_dcap_ql_wrapper.h"
#include "sgx_pce.h"
#include "sgx_error.h"

bool prepare_for_quote_generation(sgx_target_info_t* qe_target_info)
{

    int ret = 0;
    quote3_error_t qe3_ret = SGX_QL_SUCCESS;
    // We are only considering the in-proc mode and the Ubuntu setup!

    // There 2 modes on Linux: one is in-proc mode, the QE3 and PCE are loaded within the user's process. 
    // the other is out-of-proc mode, the QE3 and PCE are managed by a daemon. If you want to use in-proc
    // mode which is the default mode, you only need to install libsgx-dcap-ql. If you want to use the 
    // out-of-proc mode, you need to install libsgx-quote-ex as well. This sample is built to demo both 2
    // modes, so you need to install libsgx-quote-ex to enable the out-of-proc mode.
    // Following functions are valid in Linux in-proc mode only.
    printf("sgx_qe_set_enclave_load_policy is valid in in-proc mode only and it is optional: the default enclave load policy is persistent: \n");
    printf("set the enclave load policy as persistent:");
    
    qe3_ret = sgx_qe_set_enclave_load_policy(SGX_QL_PERSISTENT);
    if(SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in set enclave load policy: 0x%04x\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    printf("succeed!\n");

    qe3_ret = sgx_ql_set_path(SGX_QL_PCE_PATH, "/usr/lib/x86_64-linux-gnu/libsgx_pce.signed.so");
    if(SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in set PCE directory: 0x%04x.\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    
    qe3_ret = sgx_ql_set_path(SGX_QL_QE3_PATH, "/usr/lib/x86_64-linux-gnu/libsgx_qe3.signed.so");
    if(SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in set QE3 directory: 0x%04x.\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    
    qe3_ret = sgx_ql_set_path(SGX_QL_QPL_PATH, "/usr/lib/x86_64-linux-gnu/libdcap_quoteprov.so.1");
    if(SGX_QL_SUCCESS != qe3_ret) {
        printf("Info: /usr/lib/x86_64-linux-gnu/libdcap_quoteprov.so.1 not found.\n");
    }

    printf("\nGet sgx_qe_get_target_info:");
    qe3_ret = sgx_qe_get_target_info(qe_target_info);
    if (SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in sgx_qe_get_target_info. 0x%04x\n", qe3_ret);
                ret = -1;
        goto CLEANUP;
    }
    printf("succeed!");

CLEANUP:
    return ret;
}


bool create_quote(const sgx_report_t enclave_report, uint8_t** out_quote_buffer, size_t* out_quote_size)
{

    int ret = 0;
    quote3_error_t qe3_ret = SGX_QL_SUCCESS;
    uint32_t quote_size = 0;

    printf("succeed!");
    printf("\nCall sgx_qe_get_quote_size:");
    qe3_ret = sgx_qe_get_quote_size(&quote_size);
    if (SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in sgx_qe_get_quote_size. 0x%04x\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    printf("\nQuote size is: %d", quote_size);    
    *out_quote_size = quote_size;

    printf("succeed!");
    *out_quote_buffer = (uint8_t*)malloc(quote_size);
    if (NULL == *out_quote_buffer) {
        printf("Couldn't allocate quote_buffer\n");
        ret = -1;
        goto CLEANUP;
    }
    memset(*out_quote_buffer, 0, quote_size);

    // Get the Quote
    printf("\nCall sgx_qe_get_quote:");
    qe3_ret = sgx_qe_get_quote(&enclave_report,
        quote_size,
        *out_quote_buffer);
    if (SGX_QL_SUCCESS != qe3_ret) {
        printf( "Error in sgx_qe_get_quote. 0x%04x\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    printf("succeed!");

    sgx_quote3_t *p_quote;
    sgx_ql_auth_data_t *p_auth_data;
    sgx_ql_ecdsa_sig_data_t *p_sig_data;
    sgx_ql_certification_data_t *p_cert_data;
    p_quote = (sgx_quote3_t*)*out_quote_buffer;
    p_sig_data = (sgx_ql_ecdsa_sig_data_t *)p_quote->signature_data;
    p_auth_data = (sgx_ql_auth_data_t*)p_sig_data->auth_certification_data;
    p_cert_data = (sgx_ql_certification_data_t *)((uint8_t *)p_auth_data + sizeof(*p_auth_data) + p_auth_data->size);

    printf("cert_key_type = 0x%x\n", p_cert_data->cert_key_type);

CLEANUP:
    return ret;
}

bool cleanup_quote_generation()
{
    int ret = 0;
    quote3_error_t qe3_ret = SGX_QL_SUCCESS;

    printf("sgx_qe_cleanup_by_policy is valid in in-proc mode only.\n");
    printf("\n Clean up the enclave load policy:");
    qe3_ret = sgx_qe_cleanup_by_policy();
    if(SGX_QL_SUCCESS != qe3_ret) {
        printf("Error in cleanup enclave load policy: 0x%04x\n", qe3_ret);
        ret = -1;
        goto CLEANUP;
    }
    printf("succeed!\n");
    

CLEANUP:
    return ret;
}

// p_quote = (_sgx_quote3_t*)p_quote_buffer;
//     p_sig_data = (sgx_ql_ecdsa_sig_data_t *)p_quote->signature_data;
//     p_auth_data = (sgx_ql_auth_data_t*)p_sig_data->auth_certification_data;
//     p_cert_data = (sgx_ql_certification_data_t *)((uint8_t *)p_auth_data + sizeof(*p_auth_data) + p_auth_data->size);