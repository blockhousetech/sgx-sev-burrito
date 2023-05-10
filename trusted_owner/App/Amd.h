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

#include <string>
#include "../Enclave/Amd/sevapi.h"

#ifndef _AMD_H_
#define _AMD_H_

namespace amd_api_u {

    const std::string PDH_FILENAME          = "pdh.cert";      // PDH signed by PEK
    const std::string PDH_READABLE_FILENAME = "pdh_readable.txt";
    const std::string PEK_FILENAME          = "pek.cert";      // PEK signed by CEK
    const std::string PEK_READABLE_FILENAME = "pek_readable.txt";
    const std::string OCA_FILENAME          = "oca.cert";      // OCA signed by P.O.
    const std::string OCA_READABLE_FILENAME = "oca_readable.cert";
    const std::string CEK_FILENAME          = "cek.cert";      // CEK signed by ASK
    const std::string CEK_READABLE_FILENAME = "cek_readable.cert";
    const std::string ASK_FILENAME          = "ask.cert";      // ASK signed by ARK
    const std::string ASK_READABLE_FILENAME = "ask_readable.cert";
    const std::string ARK_FILENAME          = "ark.cert";      // ARK self-signed
    const std::string ARK_READABLE_FILENAME = "ark_readable.cert";

    const std::string LAUNCH_BLOB_FILENAME            = "launch_blob.bin";          // generate_launch_blob
    const std::string GUEST_OWNER_DH_FILENAME         = "godh.cert";                // generate_launch_blob
    const std::string PACKAGED_SECRET_FILENAME        = "packaged_secret.bin";      // package_secret
    const std::string PACKAGED_SECRET_HEADER_FILENAME = "packaged_secret_header.bin";// package_secret

    /**
    * Read an entire file in to a buffer, or as much as will fit.
    * Return length of file or of buffer, whichever is smaller.
    */
    size_t read_file(const std::string file_name, void *buffer, size_t len);

    /**
    * Truncate and write (not append) a file from the beginning
    * Returns number of bytes written
    */
    size_t write_file(const std::string file_name, const void *buffer, size_t len);

    /**
    * Returns the file size in number of bytes
    * May be used to tell if a file exists
    */
    size_t get_file_size(const std::string file_name);


    int import_all_certs(const std::string certs_folder, sev_certs_t *certs);

    int amd_cert_init(amd_cert *cert, const uint8_t *buffer);

}

#endif /* !_AMD_H_ */