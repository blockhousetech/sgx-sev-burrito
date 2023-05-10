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
#ifndef GRPC_server_2eproto__INCLUDED
#define GRPC_server_2eproto__INCLUDED

#include "server.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace trustedowner {

// The greeting service definition.
class TrustedOwner final {
 public:
  static constexpr char const* service_full_name() {
    return "trustedowner.TrustedOwner";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Sends a greeting
    virtual ::grpc::Status DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::trustedowner::DeployVmReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>> AsyncDeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>>(AsyncDeployVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>> PrepareAsyncDeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>>(PrepareAsyncDeployVmRaw(context, request, cq));
    }
    virtual ::grpc::Status ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::trustedowner::ProvisionVmReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>> AsyncProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>>(AsyncProvisionVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>> PrepareAsyncProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>>(PrepareAsyncProvisionVmRaw(context, request, cq));
    }
    virtual ::grpc::Status GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::trustedowner::GenerateReportForVmReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>> AsyncGenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>>(AsyncGenerateReportForVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>> PrepareAsyncGenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>>(PrepareAsyncGenerateReportForVmRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      // Sends a greeting
      virtual void DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      virtual void ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      virtual void GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() { return nullptr; }
    class async_interface* experimental_async() { return async(); }
   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>* AsyncDeployVmRaw(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::DeployVmReply>* PrepareAsyncDeployVmRaw(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>* AsyncProvisionVmRaw(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::ProvisionVmReply>* PrepareAsyncProvisionVmRaw(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>* AsyncGenerateReportForVmRaw(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::trustedowner::GenerateReportForVmReply>* PrepareAsyncGenerateReportForVmRaw(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::trustedowner::DeployVmReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>> AsyncDeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>>(AsyncDeployVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>> PrepareAsyncDeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>>(PrepareAsyncDeployVmRaw(context, request, cq));
    }
    ::grpc::Status ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::trustedowner::ProvisionVmReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>> AsyncProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>>(AsyncProvisionVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>> PrepareAsyncProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>>(PrepareAsyncProvisionVmRaw(context, request, cq));
    }
    ::grpc::Status GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::trustedowner::GenerateReportForVmReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>> AsyncGenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>>(AsyncGenerateReportForVmRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>> PrepareAsyncGenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>>(PrepareAsyncGenerateReportForVmRaw(context, request, cq));
    }
    class async final :
      public StubInterface::async_interface {
     public:
      void DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response, std::function<void(::grpc::Status)>) override;
      void DeployVm(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      void ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response, std::function<void(::grpc::Status)>) override;
      void ProvisionVm(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      void GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response, std::function<void(::grpc::Status)>) override;
      void GenerateReportForVm(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response, ::grpc::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class async* async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>* AsyncDeployVmRaw(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::trustedowner::DeployVmReply>* PrepareAsyncDeployVmRaw(::grpc::ClientContext* context, const ::trustedowner::DeployVmRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>* AsyncProvisionVmRaw(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::trustedowner::ProvisionVmReply>* PrepareAsyncProvisionVmRaw(::grpc::ClientContext* context, const ::trustedowner::ProvisionVmRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>* AsyncGenerateReportForVmRaw(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::trustedowner::GenerateReportForVmReply>* PrepareAsyncGenerateReportForVmRaw(::grpc::ClientContext* context, const ::trustedowner::GenerateReportForVmRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_DeployVm_;
    const ::grpc::internal::RpcMethod rpcmethod_ProvisionVm_;
    const ::grpc::internal::RpcMethod rpcmethod_GenerateReportForVm_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Sends a greeting
    virtual ::grpc::Status DeployVm(::grpc::ServerContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response);
    virtual ::grpc::Status ProvisionVm(::grpc::ServerContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response);
    virtual ::grpc::Status GenerateReportForVm(::grpc::ServerContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_DeployVm() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestDeployVm(::grpc::ServerContext* context, ::trustedowner::DeployVmRequest* request, ::grpc::ServerAsyncResponseWriter< ::trustedowner::DeployVmReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestProvisionVm(::grpc::ServerContext* context, ::trustedowner::ProvisionVmRequest* request, ::grpc::ServerAsyncResponseWriter< ::trustedowner::ProvisionVmReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGenerateReportForVm(::grpc::ServerContext* context, ::trustedowner::GenerateReportForVmRequest* request, ::grpc::ServerAsyncResponseWriter< ::trustedowner::GenerateReportForVmReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_DeployVm<WithAsyncMethod_ProvisionVm<WithAsyncMethod_GenerateReportForVm<Service > > > AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_DeployVm() {
      ::grpc::Service::MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::trustedowner::DeployVmRequest, ::trustedowner::DeployVmReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::trustedowner::DeployVmRequest* request, ::trustedowner::DeployVmReply* response) { return this->DeployVm(context, request, response); }));}
    void SetMessageAllocatorFor_DeployVm(
        ::grpc::MessageAllocator< ::trustedowner::DeployVmRequest, ::trustedowner::DeployVmReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::trustedowner::DeployVmRequest, ::trustedowner::DeployVmReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* DeployVm(
      ::grpc::CallbackServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithCallbackMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodCallback(1,
          new ::grpc::internal::CallbackUnaryHandler< ::trustedowner::ProvisionVmRequest, ::trustedowner::ProvisionVmReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::trustedowner::ProvisionVmRequest* request, ::trustedowner::ProvisionVmReply* response) { return this->ProvisionVm(context, request, response); }));}
    void SetMessageAllocatorFor_ProvisionVm(
        ::grpc::MessageAllocator< ::trustedowner::ProvisionVmRequest, ::trustedowner::ProvisionVmReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(1);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::trustedowner::ProvisionVmRequest, ::trustedowner::ProvisionVmReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* ProvisionVm(
      ::grpc::CallbackServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithCallbackMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodCallback(2,
          new ::grpc::internal::CallbackUnaryHandler< ::trustedowner::GenerateReportForVmRequest, ::trustedowner::GenerateReportForVmReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::trustedowner::GenerateReportForVmRequest* request, ::trustedowner::GenerateReportForVmReply* response) { return this->GenerateReportForVm(context, request, response); }));}
    void SetMessageAllocatorFor_GenerateReportForVm(
        ::grpc::MessageAllocator< ::trustedowner::GenerateReportForVmRequest, ::trustedowner::GenerateReportForVmReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(2);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::trustedowner::GenerateReportForVmRequest, ::trustedowner::GenerateReportForVmReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* GenerateReportForVm(
      ::grpc::CallbackServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/)  { return nullptr; }
  };
  typedef WithCallbackMethod_DeployVm<WithCallbackMethod_ProvisionVm<WithCallbackMethod_GenerateReportForVm<Service > > > CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_DeployVm() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_DeployVm() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestDeployVm(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodRaw(1);
    }
    ~WithRawMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestProvisionVm(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodRaw(2);
    }
    ~WithRawMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGenerateReportForVm(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_DeployVm() {
      ::grpc::Service::MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->DeployVm(context, request, response); }));
    }
    ~WithRawCallbackMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* DeployVm(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodRawCallback(1,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->ProvisionVm(context, request, response); }));
    }
    ~WithRawCallbackMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* ProvisionVm(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodRawCallback(2,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->GenerateReportForVm(context, request, response); }));
    }
    ~WithRawCallbackMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* GenerateReportForVm(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_DeployVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_DeployVm() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::trustedowner::DeployVmRequest, ::trustedowner::DeployVmReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::trustedowner::DeployVmRequest, ::trustedowner::DeployVmReply>* streamer) {
                       return this->StreamedDeployVm(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_DeployVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status DeployVm(::grpc::ServerContext* /*context*/, const ::trustedowner::DeployVmRequest* /*request*/, ::trustedowner::DeployVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedDeployVm(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::trustedowner::DeployVmRequest,::trustedowner::DeployVmReply>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_ProvisionVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_ProvisionVm() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler<
          ::trustedowner::ProvisionVmRequest, ::trustedowner::ProvisionVmReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::trustedowner::ProvisionVmRequest, ::trustedowner::ProvisionVmReply>* streamer) {
                       return this->StreamedProvisionVm(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_ProvisionVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status ProvisionVm(::grpc::ServerContext* /*context*/, const ::trustedowner::ProvisionVmRequest* /*request*/, ::trustedowner::ProvisionVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedProvisionVm(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::trustedowner::ProvisionVmRequest,::trustedowner::ProvisionVmReply>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GenerateReportForVm : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_GenerateReportForVm() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::internal::StreamedUnaryHandler<
          ::trustedowner::GenerateReportForVmRequest, ::trustedowner::GenerateReportForVmReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::trustedowner::GenerateReportForVmRequest, ::trustedowner::GenerateReportForVmReply>* streamer) {
                       return this->StreamedGenerateReportForVm(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_GenerateReportForVm() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GenerateReportForVm(::grpc::ServerContext* /*context*/, const ::trustedowner::GenerateReportForVmRequest* /*request*/, ::trustedowner::GenerateReportForVmReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGenerateReportForVm(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::trustedowner::GenerateReportForVmRequest,::trustedowner::GenerateReportForVmReply>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_DeployVm<WithStreamedUnaryMethod_ProvisionVm<WithStreamedUnaryMethod_GenerateReportForVm<Service > > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_DeployVm<WithStreamedUnaryMethod_ProvisionVm<WithStreamedUnaryMethod_GenerateReportForVm<Service > > > StreamedService;
};

}  // namespace trustedowner


#endif  // GRPC_server_2eproto__INCLUDED
