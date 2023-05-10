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
#include <fstream>
#include <stdio.h>
#include <climits>
#include "Amd.h"
#include <cstring>


namespace amd_api_u {
/**
 * Read up to len bytes from the beginning of a file
 * Returns number of bytes read, or 0 if the file couldn't be opened.
 */
size_t read_file(const std::string file_name, void *buffer, size_t len)
{
    std::ifstream file(file_name, std::ios::binary);
    if (len > INT_MAX) {
        printf("read_file Error: Input length too long\n");
        return 0;
    }
    std::streamsize slen = (std::streamsize)len;

    if (!file.is_open()) {
        printf("read_file Error: Could not open file. " \
               " ensure directory and file exists\n" \
               "  file_name: %s\n", file_name.c_str());
        return 0;
    }

    file.read((char *)buffer, slen);
    size_t count = (size_t)file.gcount();
    file.close();

    return count;
}

/**
 * Writes len bytes from the beginning of a file. Does NOT append
 * Returns number of bytes written, or 0 if the file couldn't be opened.
 * ostream CANNOT create a folder, so it has to exist already, to succeed
 */
size_t write_file(const std::string file_name, const void *buffer, size_t len)
{
    std::ofstream file(file_name, std::ofstream::out);
    if (len > INT_MAX) {
        printf("write_file Error: Input length too long\n");
        return 0;
    }
    std::streamsize slen = (std::streamsize)len;

    if (!file.is_open()) {
        printf("write_file Error: Could not open/create file. " \
               "Ensure directory exists\n" \
               "  Filename: %s\n", file_name.c_str());
        return 0;
    }
    printf("Writing to file: %s\n", file_name.c_str());

    file.write((char *)buffer, slen);
    size_t count = (size_t)file.tellp();
    file.close();

    return count;
}

/**
 * Returns the file size in number of bytes
 * May be used to tell if a file exists
 */
size_t get_file_size(const std::string file_name)
{
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        return 0;
    }

    size_t count = (size_t)file.tellg();
    file.close();

    return count;
}

int import_all_certs(const std::string certs_folder, sev_certs_t* certs)
{
    int cmd_ret = ERROR_INVALID_CERTIFICATE;
    std::string ark_full = certs_folder + ARK_FILENAME;
    std::string ask_full = certs_folder + ASK_FILENAME;
    std::string cek_full = certs_folder + CEK_FILENAME;
    std::string oca_full = certs_folder + OCA_FILENAME;
    std::string pek_full = certs_folder + PEK_FILENAME;
    std::string pdh_full = certs_folder + PDH_FILENAME;

    do {
        // Read in the ark
        uint8_t ark_buf[sizeof(amd_cert)] = {0};
        if (read_file(ark_full, ark_buf, sizeof(amd_cert)) == 0) // Variable size
            break;

        // Initialize the ark
        cmd_ret = amd_cert_init(&certs->ark, ark_buf);
        if (cmd_ret != STATUS_SUCCESS)
            break;

        // Read in the ark
        uint8_t ask_buf[sizeof(amd_cert)] = {0};
        if (read_file(ask_full, ask_buf, sizeof(amd_cert)) == 0) // Variable size
            break;

        // Initialize the ark
        cmd_ret = amd_cert_init(&certs->ask, ask_buf);
        if (cmd_ret != STATUS_SUCCESS)
            break;

        // Read in the cek
        if (read_file(cek_full, &certs->cek, sizeof(sev_cert)) != sizeof(sev_cert))
            break;

        // Read in the oca
        if (read_file(oca_full, &certs->oca, sizeof(sev_cert)) != sizeof(sev_cert))
            break;

        // Read in the pek
        if (read_file(pek_full, &certs->pek, sizeof(sev_cert)) != sizeof(sev_cert))
            break;

        // Read in the pdh
        if (read_file(pdh_full, &certs->pdh, sizeof(sev_cert)) != sizeof(sev_cert))
            break;

        cmd_ret = STATUS_SUCCESS;
    } while (0);

    return (int)cmd_ret;
}

/**
 * Initialize an amd_cert object from a (.cert file) buffer
 *
 * Parameters:
 *     cert     [out] AMD certificate object,
 *     buffer   [in]  buffer containing the raw AMD certificate
 */
int amd_cert_init(amd_cert *cert, const uint8_t *buffer)
{
    SEV_ERROR_CODE cmd_ret = STATUS_SUCCESS;
    amd_cert tmp;
    uint32_t fixed_offset = offsetof(amd_cert, pub_exp);    // 64 bytes
    uint32_t pub_exp_offset = fixed_offset;                 // 64 bytes
    uint32_t modulus_offset = 0;                            // 2k or 4k bits
    uint32_t sig_offset = 0;                                // 2k or 4k bits

    do {
        if (!cert || !buffer) {
            cmd_ret = ERROR_INVALID_PARAM;
            break;
        }

        memset(&tmp, 0, sizeof(tmp));

        // Copy the fixed body data from the temporary buffer
        memcpy(&tmp, buffer, fixed_offset);

        modulus_offset = pub_exp_offset + (tmp.pub_exp_size/8);
        sig_offset = modulus_offset + (tmp.modulus_size/8);     // Mod size as def in spec

        // Initialize the remainder of the certificate
        memcpy(&tmp.pub_exp, (void *)(buffer + pub_exp_offset), tmp.pub_exp_size/8);
        memcpy(&tmp.modulus, (void *)(buffer + modulus_offset), tmp.modulus_size/8);
        memcpy(&tmp.sig, (void *)(buffer + sig_offset), tmp.modulus_size/8);

        memcpy(cert, &tmp, sizeof(*cert));
    } while (0);

    return cmd_ret;
}
}