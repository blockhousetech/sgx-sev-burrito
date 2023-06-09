<!--
Burrito
Copyright (C) 2023 The Blockhouse Technology Limited (TBTL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->
# Burrito

Flexible remote attestation of pre-SNP SEV guest machines using SGX enclaves.

This project is a research prototype which is provided "as is" and without any sort of warranty, unless otherwise stated.

## Tamarin model

It contains a Dolev Yao model of our protocol and proofs of several security properties of it.

## Trusted owner

This folder contains the implementation of our trusted owner using the SGX SDK. Broadly speaking, the host application is a grpc server which embeds our trusted owner enclave with its three ecalls to deploy, provision, and generate report.

We have repurposed some of the code used by the [sev-tool](https://github.com/AMDESE/sev-tool) to create both a trusted and untrusted used by our trusted owner app and enclave. These libraries perform some operations that are related to the management of a SEV platform and guest VM.

We describe in a little more detail the operation of grpc methods and corresponding ecalls next:

The GRPC service definition can be found in App/proto/server.proto.

### Deploy vm call

In: SEV cert chain + SEV VM launch info
Out: SEV session buffer + SEV guest owner DH public key (GODH). For more details of these outputs check the LAUNCH_START command [here](https://www.amd.com/system/files/TechDocs/55766_SEV-KM_API_Specification.pdf).


The host application parses the inputs and pass them on to the *deploy_vm_ecall*. This ecall: 
- validates the certificate chain;
- stores launch information in its trusted state --- it includes the cek public key that is used as an identifier for the SEV platform being used;
- generates the session buffer and GODH.
    - this step involves generating the cryptographic material used to establish a secure channel between the trusted owner and the SEV VM to be deployed. All this information is created by enclave code and stores in its trusted state.
When ecall is over, the host application creates a reply containing session buffer and GODH.

### Provision vm

In: SEV launch measurement + SEV mnonce
Out: SEV secret header + SEV secret blob

For more details of these inputs and outputs check the LAUNCH_MEASURE and LAUNCH_SECRET command [here](https://www.amd.com/system/files/TechDocs/55766_SEV-KM_API_Specification.pdf).

The host application parses the inputs and pass them on to the *ecall_provision_vm*. 
This ecall:
- checks that the input measurement is correct given the launch information passed and mnonce;
- If so, it generates the secret header and secret blob.
    - the secret blob includes the CIK key used to establish an authenticated channel between trusted owner and the deployed SEV VM --- CIK is generated as part of this process. This key and the overall secret blob is generated by enclave code and stored in its trusted state.
The host application creates a reply with the expected outputs.


### Generate report for vm

In: SEV VM report data + SEV VM report data hmac
Out: SEV VM quote

The host application parses the inputs and pass them on to the *ecall_generate_report_for_vm*. 
This ecall:
- checks that the input vm data hmac matches with the expect mac for the input vm data calculated with CIK;
- If so, it generates the an SGX report containing as the report data a hash of:
    - cek public key --- identifier of the SEV platform;
    - launch info --- launch context for the SEV VM: includes SEV VM firmware digest amongst other things;
    - the input VM report data.
Given the SGX report output, the host application creates a quote using the SGX DCAP framework. This SGX (SEV) VM quote is packaged into a reply. 

This quote can be checked like any DCAP quote. The report data must be checked as per creation above, and we expect the right MRENCLAVE for the trusted owner. 

### Building

To make the trusted owner in pre-release mode:

```
make SGX_MODE=HW SGX_DEBUG=0 SGX_PRERELEASE=1
```

We use the docker container defined by Dockerfile and `docker-compose.yml` in the `.devcontainer` folder to set up our SGX building environment.

### Running

To run the trusted owner:

```
 ./trustowner
 ```

To set up a SGX host to **run** the trusted owner follow the instructions to install the DCAP driver and the software installations for a Intel SGX Application User [here](https://download.01.org/intel-sgx/sgx-dcap/1.14/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf).

Some import configuration information for the SGX platform running the trusted owner:

#### PCCS conf

The PCCS service can be installed (in the host SGX machine) with the library sgx-dcap-pccs and it needs a subscription to: https://api.portal.trustedservices.intel.com/provisioning-certification --- a key is obtained and used to configure your PCCS service. More information at: https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/QuoteGeneration/pccs.

You can place your key at the pccs config file located typically at: /opt/intel/sgx-dcap-pccs/config/default.json

Another configuration files is:

<details>
  <summary>sgx_default_qcnl.conf</summary>

```
cat /etc/sgx_default_qcnl.conf 
{
  // *** ATTENTION : This file is in JSON format so the keys are case sensitive. Don't change them.
  
  //PCCS server address
  "pccs_url": "https://localhost:8081/sgx/certification/v3/",

  // To accept insecure HTTPS certificate, set this option to false
  "use_secure_cert": false,

  // You can use the Intel PCS or another PCCS to get quote verification collateral.  Retrieval of PCK 
  // Certificates will always use the PCCS described in PCCS_URL.  When COLLATERAL_SERVICE is not defined, both 
  // PCK Certs and verification collateral will be retrieved using PCCS_URL  
  //"collateral_service": "https://api.trustedservices.intel.com/sgx/certification/v3/",

  // If you use a PCCS service to get the quote verification collateral, you can specify which PCCS API version is to be used.
  // The legacy 3.0 API will return CRLs in HEX encoded DER format and the sgx_ql_qve_collateral_t.version will be set to 3.0, while
  // the new 3.1 API will return raw DER format and the sgx_ql_qve_collateral_t.version will be set to 3.1. The PCCS_API_VERSION 
  // setting is ignored if COLLATERAL_SERVICE is set to the Intel PCS. In this case, the PCCS_API_VERSION is forced to be 3.1 
  // internally.  Currently, only values of 3.0 and 3.1 are valid.  Note, if you set this to 3.1, the PCCS use to retrieve 
  // verification collateral must support the new 3.1 APIs.
  //"pccs_api_version": "3.1",

  // Maximum retry times for QCNL. If RETRY is not defined or set to 0, no retry will be performed.
  // It will first wait one second and then for all forthcoming retries it will double the waiting time.
  // By using RETRY_DELAY you disable this exponential backoff algorithm
  "retry_times": 6,

  // Sleep this amount of seconds before each retry when a transfer has failed with a transient error
  "retry_delay": 10,

  // If LOCAL_PCK_URL is defined, the QCNL will try to retrieve PCK cert chain from LOCAL_PCK_URL first,
  // and failover to PCCS_URL as in legacy mode.
  //"local_pck_url": "http://localhost:8081/sgx/certification/v3/",

  // If LOCAL_PCK_URL is not defined, the QCNL will cache PCK certificates in memory by default.
  // The cached PCK certificates will expire after PCK_CACHE_EXPIRE_HOURS hours.
  "pck_cache_expire_hours": 168

  // You can add custom request headers and parameters to the get certificate API.
  // But the default PCCS implementation just ignores them. 
  //,"custom_request_options" : {
  //  "get_cert" : {
  //    "headers": {
  //      "head1": "value1"
  //    },
  //    "params": {
  //      "param1": "value1",
  //      "param2": "value2"
  //    }
  //  }
  //}
}
```
</details>



## SEV machine

We have created a SEV VM based off an Alpine Linux basic machine.

We create a VM using the QEMU direct boot process. It requires a kernel, an initramfs (initrd), and a kernel command line.

### Setup

Set up a SEV host and QEMU according as per instructions [here](https://github.com/AMDESE/AMDSEV/tree/e9415c071ba333a8843cdcdda8cdf0cc7cc2b9a8).

### Configuration file

There is a configuration file `.env` supplied in the root folder of the direction. One can run `source .env` to source it.

### Kernel build (sev_vm_kernel_build)

We have updated the kernel of an Alpine machine to the 5.19 version. The files to generate the Alpine package for this kernel are in folder `sev_vm_kernel_build`.

The main reason to use this kernel is the inclusion of the `efi_secret` module (https://www.phoronix.com/news/Linux-5.19-EFI-Secrets-CoCo) that allows one to retrive secrets passed to the SEV VM in a predefine secrets table.

To build the package:

```
DOCKER_BUILDKIT=1 docker build . -t alpine-kernel-5.19 -f custom-sev-kernel-5.19.Dockerfile -o .
```

The built kernel package `linux-virt-5.19-r0.apk` is output to the folder where the command was run.

A pre-built kernel can be found at `sev_orchestrator/deps/vmlinuz-virt-5.19`.


### Base Alpine VM and initramfs/initrd (sev_vm_initrd_build)

A pre-built basic Alpine VM with kernel 5.19 for a SEV host can be found at (ADD LINK).

This Alpine machine has very few packages installed, one of which is docker. It has been given a new service defined in `evaluation/service/tf_service`. It is an OpenRC service installed in the machine. This service executes the script in the VM at `/root/tf_run.sh`. At the moment, this script is as per `evaluation/service/tf_run.sh`. It imports a docker image at `/root/tf-image.tar` and creates and executes the corresponding container. This architecture was created to fit with our evaluation where are are planning to execute a container with a ML script but it is flexible and could be adapted to a number of other scenarios.

To create an initramfs/initrd with a given rootfs we do the following:
1. Harvest the QEMU disk. One can do that with the help of a script such as `evaluation/tensorflow-image/harvest_disk.sh`.
2. Use the script `update-initramfs`. Run:
```
sudo ./update-initramfs ${ALPINE_BASE_INITRAMFS} ${ALPINE_BASE_ROOTFS} initramfs-init initramfs-tf-base
```
It takes as arguments a initramfs to be modified, a rootfs to be added, and a new initramfs init script, and the name for the new initramfs, which is created on the same difectory where the command is run.

### Kernel command line

Held in `.env` file under `CMDLINE` variable.

### Boot process & attestation

Given a kernel, initrd, and kernel command line, the QEMU command that we execute to create our VM is as follows.

```
/usr/local/bin/qemu-system-x86_64 -enable-kvm -cpu EPYC -machine q35 -smp 8,maxcpus=64 -m 32000M,slots=5,maxmem=56G \
      -drive if=pflash,format=raw,unit=0,file={OVMF_FILE},readonly \
      -qmp tcp::5503,server,nowait \
      -S \
      -netdev user,id=vmnic -device e1000,netdev=vmnic,romfile= \
      -machine memory-encryption=sev0,vmport=off \
      -object sev-guest,id=sev0,cbitpos=47,reduced-phys-bits=1,session-file=data/tmp/launch_blob.base64,dh-cert-file=data/tmp/godh.base64,kernel-hashes=on \
      -kernel {KERNEL_FILE} \
      -initrd {INITRD} \
      -append "{CMDLINE}" \
      -vnc :3 -monitor pty -monitor unix:monitor,server,nowait -daemonize
```

The firmware we use --- the pre-built firmware (OVMF.fd) can be found at `sev_orchestrator/deps/OVMF.fd` --- can take into account hashes for the kernel, initrd, and kernel command line. So, once the above command is executed the firmware will load all of these elements and check their measurement. Because these measurements are added to the firmaware at VM creation time, it becomes part of the VM's (firmware) digest and measurement. In other words, the hashes of these elements are part of the digest of the VM. The `OVMF_FILE` points to the OVMF.fd file and `INITRD` variable is the new initramfs created earlier.

## SEV report generation

This folder contains two python scripts:

*sev-report-generation.py* is essentially a grpc client that calls the generate report for vm call. Given some input and the location of the CIK key, it creates a request with the appropriate VM data and VM data hmac using CIK. This script has been tailored to work with our evaluation scenarios so it takes into account two files (stdout and model) for which a hash is calculated as the VM report data. The returned quote is saved in the file `quote.dat`.

In the context of our current machine, this script is placed within the VM and used by it to contact the trusted owner in the process of creating a VM report.

It can be executed as follows, for instance:
```
python3 sev-report-generation.py --stdout-file stdout.test --model-file model.test
```

*sev-report-data.py* is a utility file and can be used to calculate the VM data for the stdout and model files.

## SEV orchestrator

Run `./download_qemu.sh` inside `sev_orchestrator/deps` folder.

For the sake of convenience, we have provided certificates and even an (throwaway/temp) OCA key pair in `sev_orchestrator/data/certs` folder. One can regenerate that as per https://github.com/AMDESE/sev-tool.

The script *sev-orchestrator.py* is responsible for orchestrating the deployment of a SEV VM according to our protocol. It is meant to be executed on the SEV host and it acts as a client to the trusted deployer and it also automates the deployment and provisioning of the SEV VM with the information returned by the trusted owner.

1. Given some input initrd, passed as a parameter to the script, it calls *deploy vm* on the trusted owner. It uses the session buffer and GODH --- together with the pre-built kernel, initrd passed as parameter, and fixed kernel command line --- to deploy a SEV VM.
2. It calls *provision vm* to get the secret blob and secret header which are used to provision the VM just deployed. Once the provisioning step is complete, the VM is started.

As the machine starts, it will then mount the rootfs inside the initrd and start the VM with that. For example:


```
sudo python3 sev-orchestrator.py --initrd /path-to/burrito/evaluation/tensorflow-image/beginner_with_secret.py_initramfs
```

In this folder, there is also a file to calculate the digest of a SEV VM for a given initrd. It assumes the kernel to be the one pre-built and the fixed kernel command line.


## VM report verifier

The vm report verifier is a small binary that carries out the verification of a VM quote, which is a SGX quote generated for a SEV VM as prescribed by our protocol. It takes in three arguments. The Mrenclave of the trusted owner --- this can be fixed once the implementation is completely solidified ---, the vm data expected, and a folder containing the "cek.cert" and "info.dat" --- the former is a AMD certificate part of the SEV chain of trust, the latter is a file generated by the trusted owner containing the launch information used in the deployment of the SEV VM in question. The two initial arguments must be provided in base64.

```
./verifier NmhlmOYt959Q5GlrR04DqDm1U6qdU8C7l1pMiC6TM38= BrLEQVYOvxSwc2rofibKM+2NabGaIAF7kPvV0LSSmCcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA== /path-to/burrito/trusted_owner/out/
```

## Evaluation

The evaluation folder has two subfolders:

- service: it contains the elements that are part of our VM's execution model. That is, the definition of the service that is called when the VM is started and the script that actually outline the behaviour of the service. At the moment, this script is executed as soon as the VM starts. The `tf_run.sh` script imports a docker image from `tf-image.tar` and executes the containers associated with it --- the securefs with the VM's CIK is passed as a volume to this container.
- tensorflow-image: it contains the steps to create the `tf-image.tar` to be executed by the `tf_service`: read build-image.README. At the moment, this image is being constructed out of a standard tensorflow container. The container executes the `go.sh` script which, in turn, executes a tensorflow script --- generating a trained model --- and it captures the stdout of the tensorflow execution. It then goes on to generate a VM quote with the measurement of these files (stdout, VM lifetime, and model.tar.gz).
    - The subfolder `report_generation` contains some files, for VM report generation, to be imported into the image. This files need to be *lazily* updated by running the `update.sh` script there. 
    - `create_initrd.sh` is a script that implements the entire pipeline of initrd generation. Given a tensorflow script, it generates a docker image, exports it into a `tf-image.tar` file, replaces this file into `/root/tf-image.tar` in the `alpine-tf-base.qcow2` image, harvest this modified rootfs to create `alpine-tf-base.tar.gz`, and uses the `update_initramfs`script to create a new initrd with this rootfs.

## Examples

Download the VM by executing `download-qcow.sh` and run `./download_qemu.sh` inside `sev_orchestrator/deps` folder. Setup openssl temporary keys `throwaway` and `throwaway.pub` in `evaluation/tensorflow-images` and install `throwaway.pub` in the SEV host so that the VM can `scp` to the SEV host.

We provide all the elements that we used in creating our VMs such as the alpine base image disk `alpine-tf-base.qcow2`.

There is an `examples` folder contains the examples that we used in our evaluation and a helper script for running them. To execute an example: 
1. Start the trusted owner on the SGX machine; 
2. Run `run-example.sh` on the SEV host with an appropriate python script as a parameter. For instance,
```
./run-example.sh beginner.py
```
This will run the whole evaluation process and output the relevant files in the SEV host in the folder `~/eval-tf/` files. For instance, for the example above it generates: `beginner.py_model.tar.gz` (tensor flow model), `beginner.py_quote.dat` (quote), `beginner.py_report_time.txt` (vm quote generation time), `beginner.py_stdout.txt` (stdout).

## Citing

Burrito is a part of ~~Mexican cuisine~~ the research effort that we present in [[1]](#1); the preprint is available [here](https://arxiv.org/abs/2305.09351). If you want to refer to our work, please use the following BibTeX entry for citation.

```
 @article{Antonino-Derek-Woloszyn23:Flexible-remote-attestation-of-pre-SNP-SEV-VMs-using-SGX-enclaves,
      title={Flexible remote attestation of pre-SNP SEV VMs using SGX enclaves}, 
      author={Pedro Antonino and Ante Derek and Wojciech Aleksander Wołoszyn},
      year={2023},
      eprint={2305.09351},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
 }
```

## References

<a id="1">[1]</a> Pedro Antonino and Ante Derek and Wojciech Aleksander Wołoszyn (2023). 
Flexible remote attestation of pre-SNP SEV VMs using SGX enclaves. Submitted with preprint
available at [https://arxiv.org/abs/2305.09351](https://arxiv.org/abs/2305.09351).





