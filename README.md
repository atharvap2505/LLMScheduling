# Issues:

## 18/10/24
- Unable to test the model at all, I've tried running it but either it takes forever, or my data limit on BITS Wi-Fi gets exhausted, probably due to the huge model size for 70B parameters.
- The code looks convincing enough, but for added confidence will require actual testing.

## 20/10/24
- Updated the code to now automate GPU allocation, and now supports having multiple GPUs.
- No need to manually load the model to GPUs anymore, it is automated by `load_check_and_dispatch` from accelerate library.
- Still haven't tested the code, so again, unsure if it'll actually work.

## 31/10/24
- Added Docker containers, values for which can be fetched from a .env file.
- Used the DeepSpeed package to implement pipeline parallelism.
- Added more optimizations such as fp16 and ZeRO based optimization.
- Requires testing to check if it actually works.
