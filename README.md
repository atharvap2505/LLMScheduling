# Issues:

## 18/10/24
- Unable to test the model at all, I've tried running it but either it takes forever, or my data limit on BITS Wi-Fi gets exhausted, probably due to the huge model size for 70B parameters.
- The code looks convincing enough, although it's majorly AI generated. I've tried understanding what each line does but I'm still underconfident due to lack of testing.
- I understand what the code is trying to do, the "how" is what will take some time to figure out (thankfully it's Python so I'll be fine)

## 20/10/24
- Updated the code to now automate GPU allocation, and now supports having multiple GPUs.
- No need to manually load the model to GPUs anymore, it is automated by `load_check_and_dispatch` from accelerate library.
- Still haven't tested the code, so again, unsure if it'll actually work.

