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
#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "Amd.h"

#include "server.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using trustedowner::TrustedOwner;
using trustedowner::DeployVmRequest;
using trustedowner::DeployVmReply;
using trustedowner::ProvisionVmRequest;
using trustedowner::ProvisionVmReply;
using trustedowner::GenerateReportForVmRequest;
using trustedowner::GenerateReportForVmReply;

std::string read_cert(const std::string& filename, size_t len) {
  char buf[len] = {0};
  amd_api_u::read_file(filename, buf, len);
  return std::string(buf, len);
}

class TrustedOwnerClient {
 public:
  TrustedOwnerClient(std::shared_ptr<Channel> channel)
      : stub_(TrustedOwner::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  void DeployVm() {
    // Data we are sending to the server.
    DeployVmRequest request;

    std::string certs_folder = "certs/";

    trustedowner::DeployVmRequest_AmdCerts* certs = request.mutable_certs();
    certs->set_ark(read_cert(certs_folder + amd_api_u::ARK_FILENAME, sizeof(amd_cert)));
    certs->set_ask(read_cert(certs_folder + amd_api_u::ASK_FILENAME, sizeof(amd_cert)));
    certs->set_cek(read_cert(certs_folder + amd_api_u::CEK_FILENAME, sizeof(sev_cert)));
    certs->set_oca(read_cert(certs_folder + amd_api_u::OCA_FILENAME, sizeof(sev_cert)));
    certs->set_pek(read_cert(certs_folder + amd_api_u::PEK_FILENAME, sizeof(sev_cert)));
    certs->set_pdh(read_cert(certs_folder + amd_api_u::PDH_FILENAME, sizeof(sev_cert)));

    trustedowner::DeployVmRequest_MeasurementInfo* info = request.mutable_info();
    info->set_api_major(1);
    info->set_api_minor(2);
    info->set_build_id(3);
    info->set_policy(4);
    info->set_digest({0});


    // Container for the data we expect from the server.
    DeployVmReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->DeployVm(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      std::cout << reply.DebugString() << std::endl;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

  void ProvisionVm() {
    // Data we are sending to the server.
    ProvisionVmRequest request;
    request.set_measurement({0});
    request.set_mnonce({0});

    // Container for the data we expect from the server.
    ProvisionVmReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->ProvisionVm(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
       std::cout << reply.DebugString() << std::endl;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

  void GenerateReportForVm() {
    // Data we are sending to the server.
    GenerateReportForVmRequest request;
    request.set_vm_data({0});
    request.set_vm_data_hmac({0});

    // Container for the data we expect from the server.
    GenerateReportForVmReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->GenerateReportForVm(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      std::cout << reply.DebugString() << std::endl;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

 private:
  std::unique_ptr<TrustedOwner::Stub> stub_;
};

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint specified by
  // the argument "--target=" which is the only expected argument.
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).
  std::string target_str;
  std::string arg_str("--target");
  if (argc > 1) {
    std::string arg_val = argv[1];
    size_t start_pos = arg_val.find(arg_str);
    if (start_pos != std::string::npos) {
      start_pos += arg_str.size();
      if (arg_val[start_pos] == '=') {
        target_str = arg_val.substr(start_pos + 1);
      } else {
        std::cout << "The only correct argument syntax is --target="
                  << std::endl;
        return 0;
      }
    } else {
      std::cout << "The only acceptable argument is --target=" << std::endl;
      return 0;
    }
  } else {
    target_str = "localhost:50051";
  }
  TrustedOwnerClient trustedOwnerClient(
      grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
  trustedOwnerClient.DeployVm();
  trustedOwnerClient.ProvisionVm();
  trustedOwnerClient.GenerateReportForVm();

  return 0;
}