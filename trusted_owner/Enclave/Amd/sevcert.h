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

#ifndef SEVCERT_H
#define SEVCERT_H

#include "sevapi.h"
#include <string>
#include <openssl/evp.h>

// Public global functions
static std::string sev_empty = "NULL";

class SEVCert {
private:
    SEV_ERROR_CODE validate_usage(uint32_t Usage);
    SEV_ERROR_CODE validate_rsa_pub_key(const sev_cert *cert,
                                        const EVP_PKEY *PublicKey);
    SEV_ERROR_CODE validate_public_key(const sev_cert *cert,
                                       const EVP_PKEY *PublicKey);
    SEV_ERROR_CODE validate_signature(const sev_cert *child_cert,
                                      const sev_cert *parent_cert,
                                      EVP_PKEY *parent_signing_key);
    SEV_ERROR_CODE validate_body(const sev_cert *cert);

    sev_cert *m_child_cert;

public:
    SEVCert(sev_cert *cert) { m_child_cert = cert; }
    ~SEVCert() {};

    const sev_cert *data() { return m_child_cert; }

    bool create_godh_cert(EVP_PKEY **godh_key_pair,
                          uint8_t api_major,
                          uint8_t api_minor);
    bool create_oca_cert(EVP_PKEY **oca_key_pair,
                         SEV_SIG_ALGO algo);
    bool sign_with_key(uint32_t version, uint32_t pub_key_usage,
                       uint32_t pub_key_algorithm, EVP_PKEY **priv_key,
                       uint32_t sig1_usage, const SEV_SIG_ALGO sig1_algo);
    SEV_ERROR_CODE compile_public_key_from_certificate(const sev_cert *cert,
                                                       EVP_PKEY *evp_pub_key);
    SEV_ERROR_CODE decompile_public_key_into_certificate(sev_cert *cert,
                                                         EVP_PKEY *evp_pubkey);
    SEV_ERROR_CODE verify_sev_cert(const sev_cert *parent_cert1,
                                   const sev_cert *parent_cert2 = NULL);
};

#endif /* SEVCERT_H */
