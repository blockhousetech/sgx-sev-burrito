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

#ifndef SEVCORE_H
#define SEVCORE_H

#include <string>

const std::string DEFAULT_SEV_DEVICE     = "/dev/sev";

#define AMD_SEV_DEVELOPER_SITE    "https://developer.amd.com/sev/"
#define ASK_ARK_PATH_SITE         "https://developer.amd.com/wp-content/resources/"

const std::string ASK_ARK_NAPLES_FILE    = "ask_ark_naples.cert";
const std::string ASK_ARK_ROME_FILE      = "ask_ark_rome.cert";
const std::string ASK_ARK_MILAN_FILE     = "ask_ark_milan.cert";
const std::string ASK_ARK_NAPLES_SITE    = ASK_ARK_PATH_SITE + ASK_ARK_NAPLES_FILE;
const std::string ASK_ARK_ROME_SITE      = ASK_ARK_PATH_SITE + ASK_ARK_ROME_FILE;
const std::string ASK_ARK_MILAN_SITE     = ASK_ARK_PATH_SITE + ASK_ARK_MILAN_FILE;

constexpr uint32_t NAPLES_FAMILY     = 0x17UL;      // 23
constexpr uint32_t NAPLES_MODEL_LOW  = 0x00UL;
constexpr uint32_t NAPLES_MODEL_HIGH = 0x0FUL;
constexpr uint32_t ROME_FAMILY       = 0x17UL;      // 23
constexpr uint32_t ROME_MODEL_LOW    = 0x30UL;
constexpr uint32_t ROME_MODEL_HIGH   = 0x3FUL;
constexpr uint32_t MILAN_FAMILY      = 0x19UL;      // 25
constexpr uint32_t MILAN_MODEL_LOW   = 0x00UL;
constexpr uint32_t MILAN_MODEL_HIGH  = 0x0FUL;

enum __attribute__((mode(QI))) ePSP_DEVICE_TYPE {
    PSP_DEVICE_TYPE_INVALID = 0,
    PSP_DEVICE_TYPE_NAPLES  = 1,
    PSP_DEVICE_TYPE_ROME    = 2,
    PSP_DEVICE_TYPE_MILAN   = 3,
};

/**
 * A system physical address that should always be invalid.
 * Used to test the SEV FW detects such invalid addresses and returns the
 * correct error return value.
 */
constexpr uint64_t INVALID_ADDRESS  = (0xFFF00000018); // Needs to be bigger than 0xFFCFFFFFFFF (16TB memory)
constexpr uint32_t BAD_ASID         = ((uint32_t)~0);
constexpr uint32_t BAD_DEVICE_TYPE  = ((uint32_t)~0);
constexpr uint32_t BAD_FAMILY_MODEL = ((uint32_t)~0);

// Platform Status Buffer flags param was split up into owner/ES in API v0.17
constexpr uint8_t  PLAT_STAT_OWNER_OFFSET    = 0;
constexpr uint8_t  PLAT_STAT_CONFIGES_OFFSET = 8;
constexpr uint32_t PLAT_STAT_OWNER_MASK      = (1U << PLAT_STAT_OWNER_OFFSET);
constexpr uint32_t PLAT_STAT_ES_MASK         = (1U << PLAT_STAT_CONFIGES_OFFSET);

namespace sev
{
// Global Function that doesn't require ioctls
bool min_api_version(unsigned platform_major, unsigned platform_minor,
                     unsigned api_major, unsigned api_minor);
};

#endif /* SEVCORE_H */
