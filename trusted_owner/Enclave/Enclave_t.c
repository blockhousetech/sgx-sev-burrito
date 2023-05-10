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
#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


typedef struct ms_ecall_deploy_vm_t {
	int ms_retval;
	sev_certs_t ms_certs;
	measurement_info_t ms_measurement_info;
	sev_session_buf* ms_out_session_data_buf;
	sev_cert* ms_out_godh_pubkey_cert;
} ms_ecall_deploy_vm_t;

typedef struct ms_ecall_provision_vm_t {
	int ms_retval;
	uint8_t* ms_hmac_measurement;
	nonce_128* ms_mnonce;
	sev_hdr_buf* ms_out_packaged_secret_header;
	uint8_t* ms_out_encrypted_blob;
	measurement_info_t* ms_measurement_info;
} ms_ecall_provision_vm_t;

typedef struct ms_ecall_generate_report_for_vm_t {
	int ms_retval;
	uint8_t* ms_vm_data;
	hmac_sha_256* ms_vm_data_hmac;
	sgx_target_info_t* ms_qe_target_info;
	sgx_report_t* ms_out_sgx_report;
} ms_ecall_generate_report_for_vm_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

typedef struct ms_pthread_wait_timeout_ocall_t {
	int ms_retval;
	unsigned long long ms_waiter;
	unsigned long long ms_timeout;
} ms_pthread_wait_timeout_ocall_t;

typedef struct ms_pthread_create_ocall_t {
	int ms_retval;
	unsigned long long ms_self;
} ms_pthread_create_ocall_t;

typedef struct ms_pthread_wakeup_ocall_t {
	int ms_retval;
	unsigned long long ms_waiter;
} ms_pthread_wakeup_ocall_t;

static sgx_status_t SGX_CDECL sgx_ecall_deploy_vm(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_deploy_vm_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_deploy_vm_t* ms = SGX_CAST(ms_ecall_deploy_vm_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	sev_session_buf* _tmp_out_session_data_buf = ms->ms_out_session_data_buf;
	size_t _len_out_session_data_buf = sizeof(sev_session_buf);
	sev_session_buf* _in_out_session_data_buf = NULL;
	sev_cert* _tmp_out_godh_pubkey_cert = ms->ms_out_godh_pubkey_cert;
	size_t _len_out_godh_pubkey_cert = sizeof(sev_cert);
	sev_cert* _in_out_godh_pubkey_cert = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_out_session_data_buf, _len_out_session_data_buf);
	CHECK_UNIQUE_POINTER(_tmp_out_godh_pubkey_cert, _len_out_godh_pubkey_cert);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_out_session_data_buf != NULL && _len_out_session_data_buf != 0) {
		if ((_in_out_session_data_buf = (sev_session_buf*)malloc(_len_out_session_data_buf)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_session_data_buf, 0, _len_out_session_data_buf);
	}
	if (_tmp_out_godh_pubkey_cert != NULL && _len_out_godh_pubkey_cert != 0) {
		if ((_in_out_godh_pubkey_cert = (sev_cert*)malloc(_len_out_godh_pubkey_cert)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_godh_pubkey_cert, 0, _len_out_godh_pubkey_cert);
	}
	_in_retval = ecall_deploy_vm(ms->ms_certs, ms->ms_measurement_info, _in_out_session_data_buf, _in_out_godh_pubkey_cert);
	if (MEMCPY_S(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_out_session_data_buf) {
		if (MEMCPY_S(_tmp_out_session_data_buf, _len_out_session_data_buf, _in_out_session_data_buf, _len_out_session_data_buf)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_out_godh_pubkey_cert) {
		if (MEMCPY_S(_tmp_out_godh_pubkey_cert, _len_out_godh_pubkey_cert, _in_out_godh_pubkey_cert, _len_out_godh_pubkey_cert)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_out_session_data_buf) free(_in_out_session_data_buf);
	if (_in_out_godh_pubkey_cert) free(_in_out_godh_pubkey_cert);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_provision_vm(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_provision_vm_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_provision_vm_t* ms = SGX_CAST(ms_ecall_provision_vm_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	uint8_t* _tmp_hmac_measurement = ms->ms_hmac_measurement;
	size_t _len_hmac_measurement = 32 * sizeof(uint8_t);
	uint8_t* _in_hmac_measurement = NULL;
	nonce_128* _tmp_mnonce = ms->ms_mnonce;
	size_t _len_mnonce = sizeof(nonce_128);
	nonce_128* _in_mnonce = NULL;
	sev_hdr_buf* _tmp_out_packaged_secret_header = ms->ms_out_packaged_secret_header;
	size_t _len_out_packaged_secret_header = sizeof(sev_hdr_buf);
	sev_hdr_buf* _in_out_packaged_secret_header = NULL;
	uint8_t* _tmp_out_encrypted_blob = ms->ms_out_encrypted_blob;
	size_t _len_out_encrypted_blob = 64 * sizeof(uint8_t);
	uint8_t* _in_out_encrypted_blob = NULL;
	measurement_info_t* _tmp_measurement_info = ms->ms_measurement_info;
	size_t _len_measurement_info = sizeof(measurement_info_t);
	measurement_info_t* _in_measurement_info = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_hmac_measurement, _len_hmac_measurement);
	CHECK_UNIQUE_POINTER(_tmp_mnonce, _len_mnonce);
	CHECK_UNIQUE_POINTER(_tmp_out_packaged_secret_header, _len_out_packaged_secret_header);
	CHECK_UNIQUE_POINTER(_tmp_out_encrypted_blob, _len_out_encrypted_blob);
	CHECK_UNIQUE_POINTER(_tmp_measurement_info, _len_measurement_info);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_hmac_measurement != NULL && _len_hmac_measurement != 0) {
		if ( _len_hmac_measurement % sizeof(*_tmp_hmac_measurement) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_hmac_measurement = (uint8_t*)malloc(_len_hmac_measurement);
		if (_in_hmac_measurement == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_hmac_measurement, _len_hmac_measurement, _tmp_hmac_measurement, _len_hmac_measurement)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_mnonce != NULL && _len_mnonce != 0) {
		_in_mnonce = (nonce_128*)malloc(_len_mnonce);
		if (_in_mnonce == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_mnonce, _len_mnonce, _tmp_mnonce, _len_mnonce)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_out_packaged_secret_header != NULL && _len_out_packaged_secret_header != 0) {
		if ((_in_out_packaged_secret_header = (sev_hdr_buf*)malloc(_len_out_packaged_secret_header)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_packaged_secret_header, 0, _len_out_packaged_secret_header);
	}
	if (_tmp_out_encrypted_blob != NULL && _len_out_encrypted_blob != 0) {
		if ( _len_out_encrypted_blob % sizeof(*_tmp_out_encrypted_blob) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_out_encrypted_blob = (uint8_t*)malloc(_len_out_encrypted_blob)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_encrypted_blob, 0, _len_out_encrypted_blob);
	}
	if (_tmp_measurement_info != NULL && _len_measurement_info != 0) {
		if ((_in_measurement_info = (measurement_info_t*)malloc(_len_measurement_info)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_measurement_info, 0, _len_measurement_info);
	}
	_in_retval = ecall_provision_vm(_in_hmac_measurement, _in_mnonce, _in_out_packaged_secret_header, _in_out_encrypted_blob, _in_measurement_info);
	if (MEMCPY_S(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_out_packaged_secret_header) {
		if (MEMCPY_S(_tmp_out_packaged_secret_header, _len_out_packaged_secret_header, _in_out_packaged_secret_header, _len_out_packaged_secret_header)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_out_encrypted_blob) {
		if (MEMCPY_S(_tmp_out_encrypted_blob, _len_out_encrypted_blob, _in_out_encrypted_blob, _len_out_encrypted_blob)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_in_measurement_info) {
		if (MEMCPY_S(_tmp_measurement_info, _len_measurement_info, _in_measurement_info, _len_measurement_info)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_hmac_measurement) free(_in_hmac_measurement);
	if (_in_mnonce) free(_in_mnonce);
	if (_in_out_packaged_secret_header) free(_in_out_packaged_secret_header);
	if (_in_out_encrypted_blob) free(_in_out_encrypted_blob);
	if (_in_measurement_info) free(_in_measurement_info);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_generate_report_for_vm(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_generate_report_for_vm_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_generate_report_for_vm_t* ms = SGX_CAST(ms_ecall_generate_report_for_vm_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	uint8_t* _tmp_vm_data = ms->ms_vm_data;
	size_t _len_vm_data = 64 * sizeof(uint8_t);
	uint8_t* _in_vm_data = NULL;
	hmac_sha_256* _tmp_vm_data_hmac = ms->ms_vm_data_hmac;
	size_t _len_vm_data_hmac = sizeof(hmac_sha_256);
	hmac_sha_256* _in_vm_data_hmac = NULL;
	sgx_target_info_t* _tmp_qe_target_info = ms->ms_qe_target_info;
	size_t _len_qe_target_info = sizeof(sgx_target_info_t);
	sgx_target_info_t* _in_qe_target_info = NULL;
	sgx_report_t* _tmp_out_sgx_report = ms->ms_out_sgx_report;
	size_t _len_out_sgx_report = sizeof(sgx_report_t);
	sgx_report_t* _in_out_sgx_report = NULL;
	int _in_retval;

	CHECK_UNIQUE_POINTER(_tmp_vm_data, _len_vm_data);
	CHECK_UNIQUE_POINTER(_tmp_vm_data_hmac, _len_vm_data_hmac);
	CHECK_UNIQUE_POINTER(_tmp_qe_target_info, _len_qe_target_info);
	CHECK_UNIQUE_POINTER(_tmp_out_sgx_report, _len_out_sgx_report);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_vm_data != NULL && _len_vm_data != 0) {
		if ( _len_vm_data % sizeof(*_tmp_vm_data) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_vm_data = (uint8_t*)malloc(_len_vm_data);
		if (_in_vm_data == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_vm_data, _len_vm_data, _tmp_vm_data, _len_vm_data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_vm_data_hmac != NULL && _len_vm_data_hmac != 0) {
		_in_vm_data_hmac = (hmac_sha_256*)malloc(_len_vm_data_hmac);
		if (_in_vm_data_hmac == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_vm_data_hmac, _len_vm_data_hmac, _tmp_vm_data_hmac, _len_vm_data_hmac)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_qe_target_info != NULL && _len_qe_target_info != 0) {
		_in_qe_target_info = (sgx_target_info_t*)malloc(_len_qe_target_info);
		if (_in_qe_target_info == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_qe_target_info, _len_qe_target_info, _tmp_qe_target_info, _len_qe_target_info)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_out_sgx_report != NULL && _len_out_sgx_report != 0) {
		if ((_in_out_sgx_report = (sgx_report_t*)malloc(_len_out_sgx_report)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_sgx_report, 0, _len_out_sgx_report);
	}
	_in_retval = ecall_generate_report_for_vm(_in_vm_data, _in_vm_data_hmac, _in_qe_target_info, _in_out_sgx_report);
	if (MEMCPY_S(&ms->ms_retval, sizeof(ms->ms_retval), &_in_retval, sizeof(_in_retval))) {
		status = SGX_ERROR_UNEXPECTED;
		goto err;
	}
	if (_in_out_sgx_report) {
		if (MEMCPY_S(_tmp_out_sgx_report, _len_out_sgx_report, _in_out_sgx_report, _len_out_sgx_report)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_vm_data) free(_in_vm_data);
	if (_in_vm_data_hmac) free(_in_vm_data_hmac);
	if (_in_qe_target_info) free(_in_qe_target_info);
	if (_in_out_sgx_report) free(_in_out_sgx_report);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[3];
} g_ecall_table = {
	3,
	{
		{(void*)(uintptr_t)sgx_ecall_deploy_vm, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_provision_vm, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_generate_report_for_vm, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[9][3];
} g_dyn_entry_table = {
	9,
	{
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		if (MEMCPY_S(&ms->ms_str, sizeof(const char*), &__tmp, sizeof(const char*))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (MEMCPY_S(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}

	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_cpuinfo = 4 * sizeof(int);

	ms_sgx_oc_cpuidex_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_oc_cpuidex_t);
	void *__tmp = NULL;

	void *__tmp_cpuinfo = NULL;

	CHECK_ENCLAVE_POINTER(cpuinfo, _len_cpuinfo);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (cpuinfo != NULL) ? _len_cpuinfo : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_oc_cpuidex_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_oc_cpuidex_t));
	ocalloc_size -= sizeof(ms_sgx_oc_cpuidex_t);

	if (cpuinfo != NULL) {
		if (MEMCPY_S(&ms->ms_cpuinfo, sizeof(int*), &__tmp, sizeof(int*))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp_cpuinfo = __tmp;
		if (_len_cpuinfo % sizeof(*cpuinfo) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		MEMSET(__tmp_cpuinfo, 0, _len_cpuinfo);
		__tmp = (void *)((size_t)__tmp + _len_cpuinfo);
		ocalloc_size -= _len_cpuinfo;
	} else {
		ms->ms_cpuinfo = NULL;
	}

	if (MEMCPY_S(&ms->ms_leaf, sizeof(ms->ms_leaf), &leaf, sizeof(leaf))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	if (MEMCPY_S(&ms->ms_subleaf, sizeof(ms->ms_subleaf), &subleaf, sizeof(subleaf))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
		if (cpuinfo) {
			if (memcpy_s((void*)cpuinfo, _len_cpuinfo, __tmp_cpuinfo, _len_cpuinfo)) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_wait_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);

	if (MEMCPY_S(&ms->ms_self, sizeof(ms->ms_self), &self, sizeof(self))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_set_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);

	if (MEMCPY_S(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_setwait_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);

	if (MEMCPY_S(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	if (MEMCPY_S(&ms->ms_self, sizeof(ms->ms_self), &self, sizeof(self))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(4, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_waiters = total * sizeof(void*);

	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(waiters, _len_waiters);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (waiters != NULL) ? _len_waiters : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_multiple_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);

	if (waiters != NULL) {
		if (MEMCPY_S(&ms->ms_waiters, sizeof(const void**), &__tmp, sizeof(const void**))) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		if (_len_waiters % sizeof(*waiters) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (MEMCPY_S(__tmp, ocalloc_size, waiters, _len_waiters)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_waiters);
		ocalloc_size -= _len_waiters;
	} else {
		ms->ms_waiters = NULL;
	}

	if (MEMCPY_S(&ms->ms_total, sizeof(ms->ms_total), &total, sizeof(total))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(5, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL pthread_wait_timeout_ocall(int* retval, unsigned long long waiter, unsigned long long timeout)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_pthread_wait_timeout_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_pthread_wait_timeout_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_pthread_wait_timeout_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_pthread_wait_timeout_ocall_t));
	ocalloc_size -= sizeof(ms_pthread_wait_timeout_ocall_t);

	if (MEMCPY_S(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	if (MEMCPY_S(&ms->ms_timeout, sizeof(ms->ms_timeout), &timeout, sizeof(timeout))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(6, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL pthread_create_ocall(int* retval, unsigned long long self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_pthread_create_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_pthread_create_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_pthread_create_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_pthread_create_ocall_t));
	ocalloc_size -= sizeof(ms_pthread_create_ocall_t);

	if (MEMCPY_S(&ms->ms_self, sizeof(ms->ms_self), &self, sizeof(self))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(7, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL pthread_wakeup_ocall(int* retval, unsigned long long waiter)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_pthread_wakeup_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_pthread_wakeup_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_pthread_wakeup_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_pthread_wakeup_ocall_t));
	ocalloc_size -= sizeof(ms_pthread_wakeup_ocall_t);

	if (MEMCPY_S(&ms->ms_waiter, sizeof(ms->ms_waiter), &waiter, sizeof(waiter))) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}

	status = sgx_ocall(8, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

