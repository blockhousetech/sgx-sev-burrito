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

#ifdef __linux__
#include "sevcore.h"

/**
 * Verify current FW is >= API version major.minor
 * Returns true if the firmware API version is at least major.minor
 * Has to be an offline comparison (can't call platform_status itself because
 *   it needs to be used in calc_measurement)
 */
bool sev::min_api_version(unsigned platform_major, unsigned platform_minor,
                          unsigned api_major, unsigned api_minor)
{
    if ((platform_major < api_major) ||
        (platform_major == api_major && platform_minor < api_minor))
        return false;
    else
        return true;
}

#endif
