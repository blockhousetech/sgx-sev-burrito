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

#ifndef CRYPTO_H
#define CRYPTO_H

#include "sevapi.h"
#include "utilities.h"

#include <openssl/evp.h>

/**
 * DIGEST
 */
#define DIGEST_SHA256_SIZE_BYTES    (256/8) // 32
#define DIGEST_SHA384_SIZE_BYTES    (384/8) // 48
#define DIGEST_SHA512_SIZE_BYTES    (512/8) // 64
typedef uint8_t DIGESTSHA256[DIGEST_SHA256_SIZE_BYTES];
typedef uint8_t DIGESTSHA384[DIGEST_SHA384_SIZE_BYTES];
typedef uint8_t DIGESTSHA512[DIGEST_SHA512_SIZE_BYTES];

typedef enum __attribute__((mode(QI))) SHA_TYPE
{
    SHA_TYPE_256 = 0,
    SHA_TYPE_384 = 1,
} SHA_TYPE;

bool gen_hmac(hmac_sha_256 *out, const hmac_key_128 key, const uint8_t *msg, size_t msg_len);

bool generate_ecdh_key_pair(EVP_PKEY **evp_key_pair, SEV_EC curve = SEV_EC_P384);

bool digest_sha(const void *msg, size_t msg_len, uint8_t *digest,
                size_t digest_len, SHA_TYPE sha_type);

bool sign_message(sev_sig *sig, EVP_PKEY **evp_key_pair, const uint8_t *msg,
                  size_t length, const SEV_SIG_ALGO algo);

#endif /* CRYPTO_H */
