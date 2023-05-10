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

#ifndef SEVAPI_H
#define SEVAPI_H

// This file puts in to C/C++ form the definitions from the SEV FW spec.
// It should remain usable purely from C
// All SEV API indices are based off of SEV API v0.17

#if __cplusplus
typedef bool _Bool;
#endif


#include <stdint.h>

// ------------------------------- //
// --- Miscellaneous constants --- //
// ------------------------------- //

// TMR (Trusted Memory Region) size required for INIT with SEV-ES enabled
#define SEV_TMR_SIZE     (1024*1024)
#define SEV_TMR_SIZE_SNP (1024*1024*2)

// NV data size required for INIT_EX.
#define SEV_NV_SIZE      (32*1024)

// Invalid Guest handle.
#define INVALID_GUEST_HANDLE    0

#define INVALID_ASID    0


// Chapter 4.3 - Command Identifiers
/**
 * SEV commands (each entry stored in a byte).
 */
typedef enum __attribute__((mode(HI))) SEV_API_COMMAND
{
    NO_COMMAND              = 0x0,

    /* SEV Platform commands */
    INIT                    = 0x01,     /* Initialize the Platform */
    SHUTDOWN                = 0x02,     /* Shut down the Platform */
    PLATFORM_RESET          = 0x03,     /* Delete the persistent Platform state */
    PLATFORM_STATUS         = 0x04,     /* Return status of the Platform */
    PEK_GEN                 = 0x05,     /* Generate a new PEK */
    PEK_CSR                 = 0x06,     /* Generate a PEK certificate signing request */
    PEK_CERT_IMPORT         = 0x07,     /* Import the signed PEK certificate */
    PDH_CERT_EXPORT         = 0x08,     /* Export the PDH and its certificate chains */
    PDH_GEN                 = 0x09,     /* Generate a new PDH and PEK signature */
    DF_FLUSH                = 0x0A,     /* Flush the data fabric */
    DOWNLOAD_FIRMWARE       = 0x0B,     /* Download new SEV FW */
    GET_ID                  = 0x0C,     /* Get the Platform ID needed for KDS */
    INIT_EX                 = 0x0D,     /* Initialize the Platform, extended */
    NOP                     = 0x0E,     /* No operation */
    RING_BUFFER             = 0x0F,     /* Enter ring buffer command mode */

    /* SEV Guest commands */
    DECOMMISSION            = 0x20,     /* Delete the Guest's SEV context */
    ACTIVATE                = 0x21,     /* Load a Guest's key into the UMC */
    DEACTIVATE              = 0x22,     /* Unload a Guest's key from the UMC */
    GUEST_STATUS            = 0x23,     /* Query the status and metadata of a Guest */
    COPY                    = 0x24,     /* Copy/move encrypted Guest page(s) */
    ACTIVATE_EX             = 0x25,     /* the Guest is bound to a particular ASID and to CCX(s) which will be
                                           allowed to run the Guest. Then Guest's key is loaded into the UMC */

    /* SEV Guest launch commands */
    LAUNCH_START            = 0x30,     /* Begin to launch a new SEV enabled Guest */
    LAUNCH_UPDATE_DATA      = 0x31,     /* Encrypt Guest data for launch */
    LAUNCH_UPDATE_VMSA      = 0x32,     /* Encrypt Guest VMCB save area for launch */
    LAUNCH_MEASURE          = 0x33,     /* Output the launch measurement */
    LAUNCH_SECRET           = 0x34,     /* Import a Guest secret sent from the Guest owner */
    LAUNCH_FINISH           = 0x35,     /* Complete launch of Guest */
    ATTESTATION             = 0x36,     /* Attestation report containing guest measurement */

    /* SEV Guest migration commands (outgoing) */
    SEND_START              = 0x40,     /* Begin to send Guest to new remote Platform */
    SEND_UPDATE_DATA        = 0x41,     /* Re-encrypt Guest data for transmission */
    SEND_UPDATE_VMSA        = 0x42,     /* Re-encrypt Guest VMCB save area for transmission */
    SEND_FINISH             = 0x43,     /* Complete sending Guest to remote Platform */
    SEND_CANCEL             = 0x44,     /* Cancel sending Guest to remote Platform */

    /* SEV Guest migration commands (incoming) */
    RECEIVE_START           = 0x50,     /* Begin to receive Guest from remote Platform */
    RECEIVE_UPDATE_DATA     = 0x51,     /* Re-encrypt Guest data from transmission */
    RECEIVE_UPDATE_VMSA     = 0x52,     /* Re-encrypt Guest VMCB save area from transmission */
    RECEIVE_FINISH          = 0x53,     /* Complete receiving Guest from remote Platform */

    /* SEV Debugging commands */
    DBG_DECRYPT             = 0x60,     /* Decrypt Guest memory region for debugging */
    DBG_ENCRYPT             = 0x61,     /* Encrypt Guest memory region for debugging */

    /* SEV Page Migration Commands */
    SWAP_OUT                = 0x70,     /* Encrypt Guest memory region for temporary storage */
    SWAP_IN                 = 0x71,     /* Reverse of SWAP_OUT */

    SEV_MAX_API_COMMAND     = SWAP_IN,

    /* SNP Platform commands */
    SNP_INIT                = 0x81,     /* Initialize the Platform */
    SNP_SHUTDOWN            = 0x82,     /* Shut down the Platform */
    SNP_PLATFORM_STATUS     = 0x83,     /* Return status of the Platform */
    SNP_DF_FLUSH            = 0x84,     /* Flush the data fabric */
    SNP_INIT_EX             = 0x85,     /* Initialize the Platform with Parameter */

    /* SNP Guest commands */
    SNP_DECOMMISSION        = 0x90,     /* Delete the Guest's SEV context */
    SNP_ACTIVATE            = 0x91,     /* Load a Guest's key into the UMC */
    SNP_GUEST_STATUS        = 0x92,     /* Query the status and metadata of a Guest */
    SNP_GCTX_CREATE         = 0x93,     /* Create a Guest context */
    SNP_GUEST_REQUEST       = 0x94,     /* Process a Guest request */
    SNP_ACTIVATE_EX         = 0x95,     /* the Guest is bound to a particular ASID and to CCX(s) which will be
                                           allowed to run the Guest. Then Guest's key is loaded into the UMC */

    /* SNP Guest launch commands */
    SNP_LAUNCH_START        = 0xA0,     /* Begin to launch a new SEV enabled Guest */
    SNP_LAUNCH_UPDATE       = 0xA1,     /* Encrypt Guest data for launch */
    SNP_LAUNCH_FINISH       = 0xA2,     /* Complete launch of Guest */

    /* SNP Debugging commands */
    SNP_DBG_DECRYPT         = 0xB0,     /* Decrypt Guest memory region for debugging */
    SNP_DBG_ENCRYPT         = 0xB1,     /* Encrypt Guest memory region for debugging */

    /* SNP Page Migration Commands */
    SNP_SWAP_OUT            = 0xC0,     /* Encrypt Guest memory region for temporary storage */
    SNP_SWAP_IN             = 0xC1,     /* Reverse of SNP_SWAP_OUT */
    SNP_PAGE_MOVE           = 0xC2,     /* Moves contents of SNP-protected pages */
    SNP_MD_INIT             = 0xC3,     /* Init the Metadata page */
    SNP_PAGE_RECLAIM        = 0xC7,     /* Clear the immutable bit on a page */
    SNP_PAGE_UNSMASH        = 0xC8,     /* Combine 512 4k pages into one 2M page in RMP */
    SNP_CONFIG              = 0xC9,     /* Set the system wide configuration values */

    SEV_LIMIT,                          /* Invalid command ID */
} SEV_API_COMMAND;

// Chapter 5.1.2 - Platform State Machine
/**
 * SEV Platform state (each entry stored in a byte).
 *
 * @UNINIT  - The Platform is uninitialized.
 * @INIT    - The Platform is initialized, but not currently managed by any guests.
 * @WORKING - The Platform is initialized, and currently managing guests.
 *
 * Allowed Platform Commands:
 * @UNINIT  - INIT, PLATFORM_RESET, PLATFORM_STATUS, DOWNLOAD_FIRMWARE, GET_ID
 * @INIT    - SHUTDOWN, PLATFORM_STATUS, PEK_GEN, PEK_CSR, PEK_CERT_IMPORT,
 *            PDH_GEN, PDH_CERT_EXPORT, DF_FLUSH, GET_ID
 * @WORKING - SHUTDOWN, PLATFORM_STATUS, PDH_GEN, PDH_CERT_EXPORT, DF_FLUSH,
 *            GET_ID
 */
typedef enum __attribute__((mode(QI))) SEV_PLATFORM_STATE
{
    SEV_PLATFORM_UNINIT  = 0,
    SEV_PLATFORM_INIT    = 1,
    SEV_PLATFORM_WORKING = 2,

    SEV_PLATFORM_LIMIT,
} SEV_PLATFORM_STATE;

/**
 * SNP Platform state
 *
 * State        Encoding    Description                     Allowed Platform Commands
 * UNINIT       0h          Platform is uninitialized       Only SNP_INIT, SNP_PLATFORM_STATUS,
 *                                                          DOWNLOAD_FIRMWARE, GET_ID
 * INIT         1h          Platform is initialized         All SNP commands except SNP_INIT,
 *                                                          DOWNLOAD_FIRMWARE
 */
typedef enum __attribute__((mode(QI))) SNP_PLATFORM_STATE
{
    SNP_PLATFORM_UNINIT       = 0,
    SNP_PLATFORM_INIT         = 1,

    SNP_PLATFORM_LIMIT,
} SNP_PLATFORM_STATE;

// Chapter 6.1.1 - GSTATE Finite State Machine
/**
 * GSTATE Finite State machine status'
 *
 * Description:
 * @UNINIT  - The Guest is uninitialized.
 * @LUPDATE - The Guest is currently being launched and plaintext data and VMCB
 *            save areas are being imported.
 * @LSECRET - The Guest is currently being launched and ciphertext data are
 *            is being imported.
 * @RUNNING - The Guest is fully launched or migrated in, and not being
 *            migrated out to another machine.
 * @SUPDATE - The Guest is currently being migrated out to another machine.
 * @RUPDATE - The Guest is currently being migrated from another machine.
 *
 * Allowed Guest Commands:
 * @UNINIT  - LAUNCH_START, RECEIVE_START
 * @LUPDATE - LAUNCH_UPDATE_DATA, LAUNCH_UPDATE_VMSA, LAUNCH_MEASURE, ACTIVATE,
 *            DEACTIVATE, DECOMMISSION, GUEST_STATUS
 * @LSECRET - LAUNCH_SECRET, LAUNCH_FINISH, ACTIVATE, DEACTIVATE, DECOMMISSION,
 *            GUEST_STATUS
 * @RUNNING - ACTIVATE, DEACTIVATE, DECOMMISSION, SEND_START, GUEST_STATUS
 * @SUPDATE - SEND_UPDATE_DATA, SEND_UPDATE_VMSA, SEND_FINISH, SEND_CANCEL,
 *            ACTIVATE, DEACTIVATE, DECOMMISSION, GUEST_STATUS
 * @RUPDATE - RECEIVE_UDPATE_DATA, RECEIVE_UDPATE_VMSA, RECEIVE_FINISH,
 *            ACTIVATE, DEACTIVATE, DECOMMISSION, GUEST_STATUS
 */
typedef enum __attribute__((mode(QI))) SEV_GUEST_STATE
{
    SEV_GUEST_UNINIT    = 0,
    SEV_GUEST_LUPDATE   = 1,
    SEV_GUEST_LSECRET   = 2,
    SEV_GUEST_RUNNING   = 3,
    SEV_GUEST_SUPDATE   = 4,
    SEV_GUEST_RUPDATE   = 5,
    SEV_GUEST_SENT      = 6,

    SEV_GUEST_STATE_LIMIT,
} SEV_GUEST_STATE;

/**
 * State            Description     Allowed Guest Commands
 * GSTATE_INIT      Initial state   SNP_LAUNCH_START, SNP_GUEST_REQUEST (VM_IMPORT)
 *                  of the guest    SNP_PAGE_RECLAIM, SNP_DECOMMISSION
 *
 * GSTATE_LAUNCH    Guest is being  SNP_GCTX_CREATE, SNP_LAUNCH_UPDATE, SNP_LAUNCH_FINISH
 *                  launched        SNP_ACTIVATE, SNP_DECOMMISSION, SNP_PAGE_RECLAIM
 *                                  SNP_PAGE_MOVE, SNP_PAGE_SWAP_OUT, SNP_PAGE_SWAP_IN,
 *                                  SNP_PAGE_UNSMASH
 *
 * GSTATE_RUNNING   Guest is        SNP_ACTIVATE, SNP_PAGE_RECLAIM, SNP_DECOMMISSION,
 *                  currently       SNP_PAGE_MOVE, SNP_PAGE_SWAP_OUT, SNP_PAGE_SWAP_IN,
 *                  running         SNP_PAGE_UNSMASH, SNP_GUEST_REQUEST
 */
typedef enum __attribute__((mode(QI))) SNP_GUEST_STATE
{
    SNP_GUEST_INIT    = 0,
    SNP_GUEST_LAUNCH  = 1,
    SNP_GUEST_RUNNING = 2,

    SNP_GUEST_STATE_LIMIT,
} SNP_GUEST_STATE;

// Chapter 4.4 - Status Codes
/**
 * SEV Error Codes (each entry stored in a byte).
 */
typedef enum __attribute__((mode(HI))) SEV_ERROR_CODE
{
    STATUS_SUCCESS                  = 0x00,
    ERROR_INVALID_PLATFORM_STATE    = 0x01,
    ERROR_INVALID_GUEST_STATE       = 0x02,
    ERROR_INVALID_CONFIG            = 0x03,
    ERROR_INVALID_LENGTH            = 0x04,
    ERROR_ALREADY_OWNED             = 0x05,
    ERROR_INVALID_CERTIFICATE       = 0x06,
    ERROR_POLICY_FAILURE            = 0x07,
    ERROR_INACTIVE                  = 0x08,
    ERROR_INVALID_ADDRESS           = 0x09,
    ERROR_BAD_SIGNATURE             = 0x0A,
    ERROR_BAD_MEASUREMENT           = 0x0B,
    ERROR_ASID_OWNED                = 0x0C,
    ERROR_INVALID_ASID              = 0x0D,
    ERROR_WBINVD_REQUIRED           = 0x0E,
    ERROR_DF_FLUSH_REQUIRED         = 0x0F,
    ERROR_INVALID_GUEST             = 0x10,
    ERROR_INVALID_COMMAND           = 0x11,
    ERROR_ACTIVE                    = 0x12,
    ERROR_HWERROR_PLATFORM          = 0x13,
    ERROR_HWERROR_UNSAFE            = 0x14,
    ERROR_UNSUPPORTED               = 0x15,
    ERROR_INVALID_PARAM             = 0x16,
    ERROR_RESOURCE_LIMIT            = 0x17,
    ERROR_SECURE_DATA_INVALID       = 0x18,

    // SNP
    ERROR_INVALID_PAGE_SIZE         = 0x19,
    ERROR_INVALID_PAGE_STATE        = 0x1A,
    ERROR_INVALID_MDATA_ENTRY       = 0x1B,
    ERROR_INVALID_PAGE_OWNER        = 0x1C,
    ERROR_AEAD_OFLOW                = 0x1D,

    ERROR_RING_BUFFER_EXIT          = 0x1F,
    ERROR_LIMIT,
} SEV_ERROR_CODE;

// ------------------------------------------------------------ //
// --- Definition of API-defined Encryption and HMAC values --- //
// ------------------------------------------------------------ //

// Chapter 2 - Summary of Keys
typedef uint8_t aes_128_key[128/8];
typedef uint8_t hmac_key_128[128/8];
typedef uint8_t hmac_sha_256[256/8];  // 256
typedef uint8_t hmac_sha_512[512/8];  // 384, 512
typedef uint8_t nonce_128[128/8];
typedef uint8_t iv_128[128/8];

// -------------------------------------------------------------------------- //
// -- Definition of API-defined Public Key Infrastructure (PKI) structures -- //
// -------------------------------------------------------------------------- //

// Appendix C.3: SEV Certificates
#define SEV_RSA_PUB_KEY_MAX_BITS    4096
#define SEV_ECDSA_PUB_KEY_MAX_BITS  576
#define SEV_ECDH_PUB_KEY_MAX_BITS   576
#define SEV_PUB_KEY_SIZE            (SEV_RSA_PUB_KEY_MAX_BITS/8)

// Appendix C.3.1 Public Key Formats - RSA Public Key
/**
 * SEV RSA Public key information.
 *
 * @modulus_size - Size of modulus in bits.
 * @pub_exp      - The public exponent of the public key.
 * @modulus      - The modulus of the public key.
 */
typedef struct __attribute__ ((__packed__)) sev_rsa_pub_key_t
{
    uint32_t    modulus_size;
    uint8_t     pub_exp[SEV_RSA_PUB_KEY_MAX_BITS/8];
    uint8_t     modulus[SEV_RSA_PUB_KEY_MAX_BITS/8];
} sev_rsa_pub_key;

/**
 * SEV Elliptical Curve algorithm details.
 *
 * @SEV_EC_INVALID - Invalid cipher size selected.
 * @SEV_EC_P256    - 256 bit elliptical curve cipher.
 * @SEV_EC_P384    - 384 bit elliptical curve cipher.
 */
typedef enum __attribute__((mode(QI))) SEV_EC
{
    SEV_EC_INVALID = 0,
    SEV_EC_P256    = 1,
    SEV_EC_P384    = 2,
} SEV_EC;

// Appendix C.3.2: Public Key Formats - ECDSA Public Key
/**
 * SEV Elliptical Curve DSA algorithm details.
 *
 * @curve - The SEV Elliptical curve ID.
 * @qx    - x component of the public point Q.
 * @qy    - y component of the public point Q.
 * @rmbz  - RESERVED. Must be zero!
 */
typedef struct __attribute__ ((__packed__)) sev_ecdsa_pub_key_t
{
    uint32_t    curve;      // SEV_EC as a uint32_t
    uint8_t     qx[SEV_ECDSA_PUB_KEY_MAX_BITS/8];
    uint8_t     qy[SEV_ECDSA_PUB_KEY_MAX_BITS/8];
    uint8_t     rmbz[SEV_PUB_KEY_SIZE-2*SEV_ECDSA_PUB_KEY_MAX_BITS/8-sizeof(uint32_t)];
} sev_ecdsa_pub_key;

// Appendix C.3.3: Public Key Formats - ECDH Public Key
/**
 * SEV Elliptical Curve Diffie Hellman Public Key details.
 *
 * @curve - The SEV Elliptical curve ID.
 * @qx    - x component of the public point Q.
 * @qy    - y component of the public point Q.
 * @rmbz  - RESERVED. Must be zero!
 */
typedef struct __attribute__ ((__packed__)) sev_ecdh_pub_key_t
{
    uint32_t    curve;      // SEV_EC as a uint32_t
    uint8_t     qx[SEV_ECDH_PUB_KEY_MAX_BITS/8];
    uint8_t     qy[SEV_ECDH_PUB_KEY_MAX_BITS/8];
    uint8_t     rmbz[SEV_PUB_KEY_SIZE-2*SEV_ECDH_PUB_KEY_MAX_BITS/8-sizeof(uint32_t)];
} sev_ecdh_pub_key;

// Appendix C.4: Public Key Formats
/**
 * The SEV Public Key memory slot may hold RSA, ECDSA, or ECDH.
 */
typedef union
{
    sev_rsa_pub_key     rsa;
    sev_ecdsa_pub_key   ecdsa;
    sev_ecdh_pub_key    ecdh;
} sev_pubkey;

// Appendix C.4: Signature Formats
/**
 * SEV Signature may be RSA or ECDSA.
 */
#define SEV_RSA_SIG_MAX_BITS        4096
#define SEV_ECDSA_SIG_COMP_MAX_BITS 576
#define SEV_SIG_SIZE                (SEV_RSA_SIG_MAX_BITS/8)

// Appendix C.4.1: RSA Signature
/**
 * SEV RSA Signature data.
 *
 * @S - Signature bits.
 */
typedef struct __attribute__ ((__packed__)) sev_rsa_sig_t
{
    uint8_t     s[SEV_RSA_SIG_MAX_BITS/8];
} sev_rsa_sig;

// Appendix C.4.2: ECDSA Signature
/**
 * SEV Elliptical Curve Signature data.
 *
 * @r    - R component of the signature.
 * @s    - S component of the signature.
 * @rmbz - RESERVED. Must be zero!
 */
typedef struct __attribute__ ((__packed__)) sev_ecdsa_sig_t
{
    uint8_t     r[SEV_ECDSA_SIG_COMP_MAX_BITS/8];
    uint8_t     s[SEV_ECDSA_SIG_COMP_MAX_BITS/8];
    uint8_t     rmbz[SEV_SIG_SIZE-2*SEV_ECDSA_SIG_COMP_MAX_BITS/8];
} sev_ecdsa_sig;

/**
 * SEV Signature may be RSA or ECDSA.
 */
typedef union
{
    sev_rsa_sig     rsa;
    sev_ecdsa_sig   ecdsa;
} sev_sig;

// Appendix C.1: USAGE Enumeration
/**
 * SEV Usage codes.
 */
typedef enum __attribute__((mode(HI))) SEV_USAGE
{
    SEV_USAGE_ARK     = 0x0,
    SEV_USAGE_ASK     = 0x13,
    SEV_USAGE_INVALID = 0x1000,
    SEV_USAGE_OCA     = 0x1001,
    SEV_USAGE_PEK     = 0x1002,
    SEV_USAGE_PDH     = 0x1003,
    SEV_USAGE_CEK     = 0x1004,
} SEV_USAGE;

// Appendix C.1: ALGO Enumeration
/**
 * SEV Algorithm cipher codes.
 */
typedef enum __attribute__((mode(HI))) SEV_SIG_ALGO
{
    SEV_SIG_ALGO_INVALID      = 0x0,
    SEV_SIG_ALGO_RSA_SHA256   = 0x1,
    SEV_SIG_ALGO_ECDSA_SHA256 = 0x2,
    SEV_SIG_ALGO_ECDH_SHA256  = 0x3,
    SEV_SIG_ALGO_RSA_SHA384   = 0x101,
    SEV_SIG_ALGO_ECDSA_SHA384 = 0x102,
    SEV_SIG_ALGO_ECDH_SHA384  = 0x103,
} SEV_SIG_ALGO;

#define SEV_CERT_MAX_VERSION    1       // Max supported version
#define SEV_CERT_MAX_SIGNATURES 2       // Max number of sig's

// Appendix C.1: SEV Certificate Format
/**
 * SEV Certificate format.
 *
 * @version       - Certificate version, set to 01h.
 * @api_major     - If PEK, set to API major version, otherwise zero.
 * @api_minor     - If PEK, set to API minor version, otherwise zero.
 * @reserved_0    - RESERVED, Must be zero!
 * @reserved_1    - RESERVED, Must be zero!
 * @pub_key_usage - Public key usage              (SEV_SIG_USAGE).
 * @pub_key_algo  - Public key algorithm          (SEV_SIG_ALGO).
 * @pub_key       - Public Key.
 * @sig_1_usage   - Key usage of SIG1 signing key (SEV_SIG_USAGE).
 * @sig_1_algo    - First signature algorithm     (SEV_SIG_ALGO).
 * @sig_1         - First signature.
 * @sig_2_usage   - Key usage of SIG2 signing key (SEV_SIG_USAGE).
 * @sig_2_algo    - Second signature algorithm    (SEV_SIG_ALGO).
 * @sig_2         - Second signature
 */
typedef struct __attribute__ ((__packed__)) sev_cert_t
{
    uint32_t     version;           // Certificate Version. Should be 1.
    uint8_t      api_major;         // Version of API generating the
    uint8_t      api_minor;         // certificate. Unused during validation.
    uint8_t      reserved_0;
    uint8_t      reserved_1;
    uint32_t     pub_key_usage;     // SEV_USAGE
    uint32_t     pub_key_algo;      // SEV_SIG_ALGO
    sev_pubkey   pub_key;
    uint32_t     sig_1_usage;       // SEV_USAGE
    uint32_t     sig_1_algo;        // SEV_SIG_ALGO
    sev_sig      sig_1;
    uint32_t     sig_2_usage;       // SEV_USAGE
    uint32_t     sig_2_algo;        // SEV_SIG_ALGO
    sev_sig      sig_2;
} sev_cert;

// Macros used for comparing individual certificates from chain
#define PEK_IN_CERT_CHAIN(x) (&((sev_cert_chain_buf *)x)->pek_cert)
#define OCA_IN_CERT_CHAIN(x) (&((sev_cert_chain_buf *)x)->oca_cert)
#define CEK_IN_CERT_CHAIN(x) (&((sev_cert_chain_buf *)x)->cek_cert)


// Appendix B.1: Certificate Format
typedef union
{
    uint8_t     short_len[2048/8];
    uint8_t     long_len[4096/8];
} amd_cert_pub_exp;

typedef union
{
    uint8_t     short_len[2048/8];
    uint8_t     long_len[4096/8];
} amd_cert_mod;

typedef union
{
    uint8_t     short_len[2048/8];
    uint8_t     long_len[4096/8];
} amd_cert_sig;

typedef enum __attribute__((mode(QI))) AMD_SIG_USAGE
{
    AMD_USAGE_ARK   = 0x00,
    AMD_USAGE_ASK   = 0x13,
} AMD_SIG_USAGE;

// Appendix B.1: AMD Signing Key Certificate Format
typedef struct __attribute__ ((__packed__)) amd_cert_t
{
    uint32_t         version;           // Certificate Version. Should be 1.
    uint64_t         key_id_0;          // The unique ID for this key
    uint64_t         key_id_1;
    uint64_t         certifying_id_0;   // The unique ID for the key that signed this cert.
    uint64_t         certifying_id_1;   // If this cert is self-signed, then equals KEY_ID field.
    uint32_t         key_usage;         // AMD_SIG_USAGE
    uint64_t         reserved_0;
    uint64_t         reserved_1;
    uint32_t         pub_exp_size;      // Size of public exponent in bits. Must be 2048/4096.
    uint32_t         modulus_size;      // Size of modulus in bits. Must be 2048/4096.
    amd_cert_pub_exp pub_exp;           // Public exponent of this key. Size is pub_exp_size.
    amd_cert_mod     modulus;           // Public modulus of this key. Size is modulus_size.
    amd_cert_sig     sig;               // Public signature of this key. Size is modulus_size.
} amd_cert;


typedef struct sev_certs_t {
    sev_cert pdh; 
    sev_cert pek; 
    sev_cert oca; 
    sev_cert cek; 
    amd_cert ask; 
    amd_cert ark;
} sev_certs_t;

typedef struct measurement_info_t {
    uint8_t  api_major;
    uint8_t  api_minor;
    uint8_t  build_id;
    uint32_t policy;    // SEV_POLICY
    uint8_t  digest[32];   // gctx_ld -> digest[SHA256_DIGEST_LENGTH]
    nonce_128 mnonce;
} measurement_info_t;



// -------------------------------------------------------------------------- //
// Definition of buffers referred to by the command buffers of SEV API commands
// -------------------------------------------------------------------------- //
// Values passed into INIT command Options field
enum __attribute__((mode(SI))) SEV_OPTIONS
{
    // Bit 0 is the SEV-ES bit
    SEV_OPTION_SEV_ES = 1 << 0,
};

// Values returned from PLATFORM_STATUS Config.ES field
enum __attribute__((mode(SI))) SEV_CONFIG
{
    // Bit 0 is the SEV-ES bit
    SEV_CONFIG_NON_ES = 0 << 0,
    SEV_CONFIG_ES     = 1 << 0,
};

/**
 * PLATFORM_STATUS Command Sub-Buffer
 * Status of the owner of the Platform (each entry stored in one byte).
 */
enum SEV_PLATFORM_STATUS_OWNER
{
    // Bit 0 is the owner, self or external..
    PLATFORM_STATUS_OWNER_SELF     = 0 << 0,
    PLATFORM_STATUS_OWNER_EXTERNAL = 1 << 0,
};

/**
 * Transport encryption and integrity keys
 * (See sev_session_buf)
 *
 * @TEK - Transport Encryption Key.
 * @TIK - Transport Integrity Key.
 */
typedef struct __attribute__ ((__packed__)) tek_tik_t
{
    aes_128_key   tek;
    aes_128_key   tik;
} tek_tik;

/**
 * LAUNCH_START/SEND_START/RECEIVE_START Session Data Buffer
 *
 * @nonce      - An arbitrary 128 bit number.
 * @wrap_tk    - The SEV transport encryption and integrity keys.
 * @wrap_iv    - 128 bit initializer vector.
 * @wrap_mac   - Session hash message authentication code.
 * @policy_mac - policy hash message authentication code.
 */
typedef struct __attribute__ ((__packed__)) sev_session_buf_t
{
    nonce_128       nonce;
    tek_tik         wrap_tk;
    iv_128          wrap_iv;
    hmac_sha_256    wrap_mac;
    hmac_sha_256    policy_mac;
} sev_session_buf;

/**
 * LAUNCH_MEASURE Measurement buffer.
 *
 * @measurement - 256 bit hash message authentication code.
 * @m_nonce     - An arbitrary 128 bit number.
 */
typedef struct __attribute__ ((__packed__)) sev_measure_buf_t
{
    hmac_sha_256    measurement;
    nonce_128       m_nonce;
} sev_measure_buf;

/**
 * LAUNCH_SECRET, SEND_UPDATE_DATA/VMSA, RECEIVE_UPDATE_DATA/VMSA
 * HDR Buffer
 */
typedef struct __attribute__ ((__packed__)) sev_hdr_buf_t
{
    uint32_t        flags;
    iv_128          iv;
    hmac_sha_256    mac;
} sev_hdr_buf;

/**
 * PDH_CERT_EXPORT/SEND_START Platform Certificate(s) Chain Buffer
 *
 * @pek_cert - Platform Endorsement Key certificate.
 * @oca_cert - Owner Certificate Authority certificate.
 * @cek_cert - Chip Endorsement Key certificate.
 */
typedef struct __attribute__ ((__packed__)) sev_cert_chain_buf_t
{
    sev_cert    pek_cert;
    sev_cert    oca_cert;
    sev_cert    cek_cert;
} sev_cert_chain_buf;

// SEND_START AMD Certificates Buffer
typedef struct __attribute__ ((__packed__)) amd_cert_chain_buf_t
{
    amd_cert    ask_cert;
    amd_cert    ark_cert;
} amd_cert_chain_buf;

#endif /* SEVAPI_H */
