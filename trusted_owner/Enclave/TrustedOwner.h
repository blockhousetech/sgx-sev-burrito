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

#ifndef _TRUSTED_OWNER_H_
#define _TRUSTED_OWNER_H_

#include <assert.h>
#include <stdlib.h>
#include "Amd/sevapi.h"
#include <sgx_utils.h>

enum trusted_owner_status_t { 
    init,
    deployed,
    provisioned
};

struct trusted_owner_state_t {
    trusted_owner_status_t status;
    measurement_info_t measurement_info;
    sev_ecdsa_pub_key cek;
    hmac_key_128 cik;
    tek_tik tks;
};

const sgx_target_info_t qe_target_info = {};


static trusted_owner_state_t g_trusted_state = {
   init, // trusted_owner_status_t status
   {},    //measurement_info_t measurement_info
   {},    // sev_ecdsa_pub_key cek
   {},    // hmac_key_128 cik
   {},    // tek_tik_t tks
};

bool calculate_report_data(const uint8_t vm_data[64], uint8_t (*sgx_report_data) [64]);

#endif /* !_TRUSTED_OWNER_H_ */
