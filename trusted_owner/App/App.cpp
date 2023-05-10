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
#include <string.h>
#include <assert.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Amd.h"
#include "QuoteGeneration.h"
#include "Enclave_u.h"

// GRPC server includes
#include <iostream>
#include <memory>
#include <string>
#include <csignal>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "server.grpc.pb.h"

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
    {
        SGX_ERROR_MEMORY_MAP_FAILURE,
        "Failed to reserve memory for the enclave.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
    fflush(stdout);
}

void print_struct_memory(unsigned char *p, size_t size) {
    printf("Info: printing struct at %02x.\n", p);
        for (unsigned int i = 0; i < size; i++)
            printf("%02x ", p[i]);
    printf("\n");
}

void RunServer();

void handler(int signum) {
    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);

    cleanup_quote_generation();
    
    printf("Info: TrustedOwner successfully returned.\n");

    printf("Enter a character before exit ...\n");
    getchar();
    exit(0);
}

/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    signal(SIGINT, handler);
    signal(SIGTERM, handler);
    signal(SIGQUIT, handler);

    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }

    prepare_for_quote_generation(&qe_target_info);

    RunServer();
}


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using trustedowner::TrustedOwner;
using trustedowner::DeployVmRequest;
using trustedowner::DeployVmReply;
using trustedowner::ProvisionVmRequest;
using trustedowner::ProvisionVmReply;
using trustedowner::GenerateReportForVmRequest;
using trustedowner::GenerateReportForVmReply;

void write_cert_file(std::string filename, const std::string& contents, size_t len){
    amd_api_u::write_file(filename, (void *)contents.c_str(), len);
}

// Logic and data behind the server's behavior.
class TrustedOwnerServiceImpl final : public TrustedOwner::Service {

    Status DeployVm(ServerContext* context, const DeployVmRequest* request,
                    DeployVmReply* reply) override {
        printf("Info: calling DeployVm.\n");
        sev_certs_t all_certs {};
        std::string certs_folder = "certs_received/";

        write_cert_file(certs_folder + amd_api_u::ARK_FILENAME, request->certs().ark(), sizeof(amd_cert));
        write_cert_file(certs_folder + amd_api_u::ASK_FILENAME, request->certs().ask(), sizeof(amd_cert));
        write_cert_file(certs_folder + amd_api_u::CEK_FILENAME, request->certs().cek(), sizeof(sev_cert));
        write_cert_file(certs_folder + amd_api_u::OCA_FILENAME, request->certs().oca(), sizeof(sev_cert));
        write_cert_file(certs_folder + amd_api_u::PEK_FILENAME, request->certs().pek(), sizeof(sev_cert));
        write_cert_file(certs_folder + amd_api_u::PDH_FILENAME, request->certs().pdh(), sizeof(sev_cert));

        int res = amd_api_u::import_all_certs(certs_folder, &all_certs);

        printf("Info: import_all_certs successful.\n");

        // typedef struct measurement_info_t {
//     uint8_t  api_major;
//     uint8_t  api_minor;
//     uint8_t  build_id;
//     uint32_t policy;    // SEV_POLICY
//     uint8_t  digest[32];   // gctx_ld -> digest[SHA256_DIGEST_LENGTH]
//     nonce_128 mnonce;
// } measurement_info_t;

        measurement_info_t measurement_info {};
        measurement_info.api_major = request->info().api_major();
        measurement_info.api_minor = request->info().api_minor();
        measurement_info.build_id = request->info().build_id();
        measurement_info.policy = request->info().policy();
        memcpy(measurement_info.digest, request->info().digest().c_str(), sizeof(measurement_info.digest));


        sev_session_buf out_session_data_buf {}; 
        sev_cert out_godh_pubkey_cert {};

        int ret;
        printf("Info: calling ecall_deploy_vm.\n");

        print_struct_memory((unsigned char*)&out_session_data_buf, sizeof(sev_session_buf));
        print_struct_memory((unsigned char*)&out_godh_pubkey_cert, sizeof(sev_cert));

        ecall_deploy_vm(global_eid, &ret, all_certs, measurement_info, &out_session_data_buf, &out_godh_pubkey_cert);

        print_struct_memory((unsigned char*)&out_session_data_buf, sizeof(sev_session_buf));
        print_struct_memory((unsigned char*)&out_godh_pubkey_cert, sizeof(sev_cert));

        std::string buf_file = "out/" + amd_api_u::LAUNCH_BLOB_FILENAME;
        std::string godh_cert_file =  "out/" + amd_api_u::GUEST_OWNER_DH_FILENAME;

        amd_api_u::write_file(buf_file, (void *)&out_session_data_buf, sizeof(sev_session_buf));
        if (amd_api_u::write_file(godh_cert_file, &out_godh_pubkey_cert, sizeof(sev_cert)) != sizeof(sev_cert))
            printf("Something is wrong with the godh_cert_file");

        reply->set_session_buffer(std::string((char *)&out_session_data_buf, sizeof(sev_session_buf)));
        reply->set_godh_cert(std::string((char *)&out_godh_pubkey_cert, sizeof(sev_cert)));

        printf("Info: call ecall_deploy_vm successful.\n");
        return Status::OK;
    }

    Status ProvisionVm(ServerContext* context, const ProvisionVmRequest* request,
                    ProvisionVmReply* reply) override {

        printf("Info: calling ProvisionVm.\n");
        int ret;

        uint8_t hmac_measurement[32] {};
        memcpy(hmac_measurement, request->measurement().c_str(), sizeof(hmac_measurement));
        nonce_128 mnonce {};
        memcpy(mnonce, request->mnonce().c_str(), sizeof(mnonce));

        sev_hdr_buf out_packaged_secret_header {};
        uint8_t out_encrypted_blob[64] {};
        measurement_info_t out_info {}; 
        
        print_struct_memory(hmac_measurement, sizeof(hmac_measurement));
        ecall_provision_vm(global_eid, &ret, hmac_measurement, &mnonce, &out_packaged_secret_header, out_encrypted_blob, &out_info);
        print_struct_memory(hmac_measurement, sizeof(hmac_measurement));

        std::string packaged_secret_file = "out/" + amd_api_u::PACKAGED_SECRET_FILENAME;
        std::string packaged_secret_header_file = "out/" + amd_api_u::PACKAGED_SECRET_HEADER_FILENAME;

        amd_api_u::write_file(packaged_secret_file, &out_encrypted_blob, 64);
        amd_api_u::write_file(packaged_secret_header_file, &out_packaged_secret_header, sizeof(out_packaged_secret_header));
        amd_api_u::write_file("out/info.dat", &out_info, sizeof(out_info));


        reply->set_secret_header(std::string((char *)&out_packaged_secret_header, sizeof(out_packaged_secret_header)));
        reply->set_secret_blob(std::string((char *)&out_encrypted_blob, 64));

        return Status::OK;
    }

    Status GenerateReportForVm(ServerContext* context, const GenerateReportForVmRequest* request,
                    GenerateReportForVmReply* reply) override {

        printf("Info: calling GenerateReportForVm.\n");
        int ret;
        uint8_t vm_data[64] {};
        memcpy(vm_data, request->vm_data().c_str(), sizeof(vm_data));
        hmac_sha_256 vm_data_hmac {};
        memcpy(vm_data_hmac, request->vm_data_hmac().c_str(), sizeof(vm_data_hmac));
        sgx_report_t out_sgx_report {}; 
        uint8_t* out_quote_buffer = NULL; 
        size_t out_quote_buffer_size = 0;
       

        print_struct_memory((unsigned char*) &out_sgx_report, sizeof(out_sgx_report));
        ecall_generate_report_for_vm(global_eid, &ret,vm_data, &vm_data_hmac, &qe_target_info, &out_sgx_report);
        print_struct_memory((unsigned char*) &out_sgx_report, sizeof(out_sgx_report));

        create_quote(out_sgx_report, &out_quote_buffer, &out_quote_buffer_size);

        if (out_quote_buffer != NULL) {
            
            printf("Quote size=%d", out_quote_buffer_size);
            print_struct_memory((unsigned char*) out_quote_buffer, out_quote_buffer_size);
            

            // TO REMOVE
            // memset(out_quote.report_body.report_data.d, 1, 64);

            amd_api_u::write_file("out/quote.dat", out_quote_buffer, out_quote_buffer_size);
            reply->set_quote(std::string((char *)out_quote_buffer, out_quote_buffer_size));

            free(out_quote_buffer);
        }
        
        return Status::OK;
    }
};

void RunServer() {
    
    std::string server_address("0.0.0.0:50051");
    TrustedOwnerServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();

}