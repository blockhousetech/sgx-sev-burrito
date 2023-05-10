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

#include "sgx_report.h"
#include "sgx_quote_3.h"


bool prepare_for_quote_generation(sgx_target_info_t* qe_target_info);

bool create_quote(const sgx_report_t enclave_report, uint8_t** out_quote_buffer, size_t* out_quote_size);

bool cleanup_quote_generation();

// p_quote = (_sgx_quote3_t*)p_quote_buffer;
//     p_sig_data = (sgx_ql_ecdsa_sig_data_t *)p_quote->signature_data;
//     p_auth_data = (sgx_ql_auth_data_t*)p_sig_data->auth_certification_data;
//     p_cert_data = (sgx_ql_certification_data_t *)((uint8_t *)p_auth_data + sizeof(*p_auth_data) + p_auth_data->size);