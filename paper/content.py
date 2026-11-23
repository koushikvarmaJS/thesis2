"""Text of the paper draft. build_paper.py turns BODY into the Word file.

Markup inside strings: **bold**, *italic*, ^superscript^, {cite:key1,key2} (keys in REFS, numbered in order of
first use), {ref:label} (a table or figure label below). Entries of BODY:
  ('h', text, level)                          heading (level 1 or 2), numbered by hand in the text
  ('p', text) / ('pc', text)                  paragraph / paragraph without first-line indent
  ('tab', label, caption, rows, widths_in, opts)   opts: note=..., wide=True (full page width), size=7.5
  ('fig', label, path, caption, opts)         path relative to thesis2/; opts: wide=True
  ('refs',)                                   acknowledgments and the reference list
Every number is taken from Jetson_KV_Report_v2.pdf (sets A-L), plots_v2/stats_v4.json and the set N/P result files
(sets M-P), and paper/v4c_numbers.json (17 Nov reruns: set O, Q and the corrected K/N/P cells >= 8K).
"""

TITLE = 'Does Compressing the KV Cache Save Energy at the Edge?'
SUBTITLE = 'Quantization and Eviction on an NVIDIA Jetson Orin Nano (draft, November 2026)'
AUTHOR_META = 'Koushik Bhupathiraju'
# template placeholder -> running head (even pages: venue left, authors right; odd pages: title left, venue right)
RUNNING_HEADS = {'WOODSTOCK’18, June, 2018, El Paso, Texas USA': 'Draft, November 2026',
                 'F. Surname et al.': 'K. Bhupathiraju et al.',
                 'Insert Your Title Here': 'KV-Cache Compression Energy at the Edge'}
AUTHORS = [
    'Koushik Bhupathiraju\nSchool of Computing and Engineering\nQuinnipiac University\nHamden, Connecticut, USA\n'
    'Koushik.Bhupathiraju@quinnipiac.edu',
    'Taskin Kocak\nSchool of Computing and Engineering\nQuinnipiac University\nHamden, Connecticut, USA',
    'Rongyu Lin\nSchool of Computing and Engineering\nQuinnipiac University\nHamden, Connecticut, USA',
]

ABSTRACT = [
    'Compressing the key-value (KV) cache is the standard way to fit long contexts into the memory of a large '
    'language model (LLM) server. Methods either store keys and values in fewer bits (quantization) or delete the '
    'tokens that seem least useful (eviction), and they are usually justified by memory savings and throughput on '
    'data-center GPUs. Their energy, too, has been measured only there, and studies disagree on whether a '
    'quantized cache saves energy or costs it; to our knowledge, it has not been measured on a memory-limited edge '
    'device. We '
    'measure it on an NVIDIA Jetson Orin Nano (8 GB of memory shared by CPU and GPU) with the board\'s own power '
    'sensor, for six small models (0.5B to 3B parameters, three families), contexts from 1K to 64K tokens, two power '
    'modes and two inference engines: llama.cpp with 8-bit and 4-bit cache formats, and PyTorch with KIVI-style '
    'quantized caches and the StreamingLLM and SnapKV eviction methods.',
    'In llama.cpp a quantized cache never saved energy: in every set, an 8-bit or 4-bit cache used more energy per '
    'request than an f16 cache, +3 to +5% at 1K tokens and +13 to +18% at 16K (+29 to +58% per generated token), '
    'while cutting the cache by 47 to 72%. Kernel profiles show why: with a '
    'quantized cache, decode attention leaves the tensor-core kernel for a CUDA-core kernel that is 2.1 to 3.9 times '
    'slower per call. Quantization pays only as capacity: it is the only way to run a 3B model at 32K or 64K tokens, '
    'or eight 16K-token sessions of a 1B model, on this board. 4-bit keys break every Qwen2.5 model (perplexity '
    '×5 to ×741); keeping keys at 8 or 16 bits restores accuracy but not energy; and a retrieval test shows that '
    'perplexity understates the damage on Llama. In PyTorch, eviction saved only 3 to 11% per request at 8K–16K '
    'tokens while losing up to 95% of the correct answers in a retrieval test when the question is not known at '
    'compression time, and quantized caches cost 28 to 79% more energy at 8K. We also report a benchmarking pitfall '
    'in Hugging Face generation that had made eviction\'s saving look several times larger (40–47%). On this class of '
    'device, the cache to use is an uncompressed one in an efficient engine whenever it fits.',
]
KEYWORDS = ('LLM inference, KV cache, quantization, eviction, energy measurement, edge computing, '
            'NVIDIA Jetson, llama.cpp')

ACK = ('We thank Prof. Rongyu Lin for the review of the first results, which led to most of the controls in this '
       'paper (the flash-attention-off baseline, net and gross energy, per-rail energy, separate prefill and decode '
       'measurements, concurrency and paired rounds).')

# ---------------------------------------------------------------- references
REFS = {
    'vaswani': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz '
               'Kaiser and Illia Polosukhin. 2017. Attention is all you need. In *Advances in Neural Information '
               'Processing Systems 30 (NeurIPS 2017)*.',
    'pope': 'Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, '
            'Jonathan Heek, Kefan Xiao, Shivani Agrawal and Jeff Dean. 2023. Efficiently scaling transformer '
            'inference. In *Proceedings of Machine Learning and Systems 5 (MLSys 2023)*.',
    'vllm': 'Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, '
            'Hao Zhang and Ion Stoica. 2023. Efficient memory management for large language model serving with '
            'PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP \'23)*.',
    'gqa': 'Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón and Sumit Sanghai. '
           '2023. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In '
           '*Proceedings of EMNLP 2023*.',
    'h2o': 'Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong '
           'Tian, Christopher Ré, Clark Barrett, Zhangyang Wang and Beidi Chen. 2023. H2O: Heavy-hitter oracle for '
           'efficient generative inference of large language models. In *Advances in Neural Information Processing '
           'Systems 36 (NeurIPS 2023)*.',
    'streamingllm': 'Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han and Mike Lewis. 2024. Efficient streaming '
                    'language models with attention sinks. In *International Conference on Learning '
                    'Representations (ICLR 2024)*.',
    'snapkv': 'Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, '
              'Patrick Lewis and Deming Chen. 2024. SnapKV: LLM knows what you are looking for before generation. '
              'In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*.',
    'kivi': 'Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen and '
            'Xia Hu. 2024. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In *Proceedings of the '
            '41st International Conference on Machine Learning (ICML 2024)*.',
    'kvquant': 'Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer '
               'and Amir Gholami. 2024. KVQuant: Towards 10 million context length LLM inference with KV cache '
               'quantization. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*.',
    'qserve': 'Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan and Song Han. 2025. '
              'QServe: W4A8KV4 quantization and system co-design for efficient LLM serving. In *Proceedings of '
              'Machine Learning and Systems 7 (MLSys 2025)*.',
    'flashattn': 'Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré. 2022. FlashAttention: Fast '
                 'and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information '
                 'Processing Systems 35 (NeurIPS 2022)*.',
    'words2watts': 'Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William '
                   'Bergeron, Jeremy Kepner, Devesh Tiwari and Vijay Gadepally. 2023. From words to watts: '
                   'Benchmarking the energy costs of large language model inference. In *IEEE High Performance '
                   'Extreme Computing Conference (HPEC 2023)*.',
    'vallecillos': 'Fernando Vallecillos-Ruiz, Giordano d\'Aloisio, Max Hort, Luca Traini, Antinisca Di Marco and '
                   'Leon Moonen. 2026. Smaller models, unexpected costs: Trade-offs in LLM quantization for '
                   'automated program repair. In *Proceedings of the 42nd IEEE International Conference on Software '
                   'Maintenance and Evolution (ICSME 2026)*. arXiv:2606.27205.',
    'powertrace': 'Jiansheng Wang, Jaskeerat Singh, Haochen Xu, Xinyi Li, Mingyuan Yan, Shaohui Liu, Yuzhuo Li, '
                  'Yize Chen and Wenqi Cui. 2026. LLM-PowerTrace: A dataset of time-resolved GPU power traces for '
                  'AI compute. arXiv:2610.38315.',
    'kelle': 'Tianhua Xia and Sai Qian Zhang. 2025. Kelle: Co-design KV caching and eDRAM for efficient LLM serving '
             'in edge computing. In *Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture '
             '(MICRO 2025)*. https://doi.org/10.1145/3725843.3756071.',
    'mzcache': 'Hongseung Yu, Minsung Kim, Jongseok Park and Kyunghan Lee. 2026. mzCache: On-device LLM memory '
               'management under multitasking. In *Proceedings of the 32nd Annual International Conference on Mobile '
               'Computing and Networking (MobiCom 2026)*. https://doi.org/10.1145/3795866.3844495.',
    'llamacpp': 'Georgi Gerganov and contributors. 2023–2026. llama.cpp: LLM inference in C/C++. '
                'https://github.com/ggml-org/llama.cpp (commit 7adf160 used here).',
    'kvpress': 'NVIDIA. 2024–2026. kvpress: LLM KV cache compression made easy. https://github.com/NVIDIA/kvpress '
               '(version 0.5.3 used here).',
    'transformers': 'Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, '
                    'Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz and others. 2020. Transformers: '
                    'State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System '
                    'Demonstrations*.',
    'pytorch': 'Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan and others. '
               '2019. PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural '
               'Information Processing Systems 32 (NeurIPS 2019)*.',
    'quanto': 'Hugging Face. 2024–2026. optimum-quanto: a PyTorch quantization backend for Optimum. '
              'https://github.com/huggingface/optimum-quanto (version 0.2.7 used here).',
    'hqq': 'Hicham Badri and Appu Shaji. 2023. Half-quadratic quantization of large machine learning models. '
           'Mobius Labs. https://mobiusml.github.io/hqq_blog/ (hqq 0.2.8 used here).',
    'needle': 'Greg Kamradt. 2023. Needle in a haystack: pressure testing LLMs. '
              'https://github.com/gkamradt/LLMTest_NeedleInAHaystack.',
    'ruler': 'Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang and '
             'Boris Ginsburg. 2024. RULER: What\'s the real context size of your long-context language models? In '
             '*Conference on Language Modeling (COLM 2024)*.',
    'wikitext': 'Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher. 2017. Pointer sentinel mixture '
                'models. In *International Conference on Learning Representations (ICLR 2017)*.',
    'qwen25': 'Qwen Team. 2024. Qwen2.5 technical report. arXiv:2412.15115.',
    'llama3': 'Aaron Grattafiori, Abhimanyu Dubey and others (Llama Team, Meta). 2024. The Llama 3 herd of models. '
              'arXiv:2407.21783.',
    'gemma3': 'Gemma Team, Google DeepMind. 2025. Gemma 3 technical report. arXiv:2503.19786.',
    'thesis': 'Koushik Bhupathiraju. 2026. Energy benchmarking of KV cache compression methods on GPU. Thesis, '
              'Quinnipiac University, Hamden, CT.',
    'repo': 'Koushik Bhupathiraju. 2026. thesis2: code and raw data for this paper. '
            'https://github.com/koushikvarmaJS/thesis2.',
}

# ---------------------------------------------------------------- body
M2 = '−'  # minus sign

BODY = [
    ('h', '1 Introduction', 1),
    ('p', 'An LLM generates text one token at a time. To avoid recomputing the whole prompt for every new token it '
          'keeps the keys and values of all earlier tokens in a KV cache {cite:vaswani,pope}. The cache grows '
          'linearly with the context: Llama-3.2-3B needs 115 KB per token in 16-bit floating point (f16), so 32K '
          'tokens take 3.8 GB, more than the model\'s own 2 GB of 4-bit weights. On an edge device whose CPU and '
          'GPU share 8 GB, the cache, not the model, decides how long a conversation can be and how many can run at '
          'once.'),
    ('p', 'Two families of methods shrink the cache. *Quantization* stores keys and values in 8, 4 or 2 bits '
          '{cite:kivi,kvquant}. *Eviction* deletes the tokens that seem least useful, either by position '
          '(StreamingLLM keeps the first and the most recent tokens {cite:streamingllm}) or by attention score '
          '(H2O, SnapKV {cite:h2o,snapkv}). Both are evaluated mostly on data-center GPUs and justified by memory '
          'and throughput. A smaller cache means fewer bytes to read for every generated token, so it is natural '
          'to expect lower energy as well. But compression is not free: quantized values must be unpacked before '
          'use, and eviction must score tokens. Whether the trade saves energy depends on what the hardware and '
          'the inference engine do with the compressed cache. Measurements on data-center GPUs disagree: Hugging '
          'Face quantized caches raised GPU energy by 22–287% on a GH200 {cite:vallecillos}, while an FP8 cache in '
          'vLLM lowered it by about 3% {cite:powertrace}. On edge hardware, energy has been simulated for a '
          'proposed accelerator {cite:kelle}, and an on-device memory manager has been evaluated on a phone '
          '{cite:mzcache}, but to our knowledge no study has measured the energy of existing compression methods '
          'on a real edge device.'),
    ('p', 'This paper measures it. We run six small models (Qwen2.5 0.5B, 1.5B and 3B, Llama 3.2 1B and 3B, '
          'Gemma 3 1B) on an NVIDIA Jetson Orin Nano Developer Kit, read the board\'s INA3221 power sensor every '
          '50 ms, and compare every compressed configuration with an uncompressed (f16) cache in the same engine, '
          'paired by round. In llama.cpp {cite:llamacpp}, the engine most edge deployments use, we test the 8-bit '
          'and 4-bit cache formats (q8_0, q4_0, q4_1) and mixed formats with different bit widths for keys and '
          'values. In PyTorch we test the two eviction methods of the earlier T4 study {cite:thesis} through '
          'NVIDIA\'s kvpress {cite:kvpress}, and KIVI-style quantized caches through transformers '
          '{cite:transformers}. We measure energy per request and per generated token, speed, power per rail, '
          'memory, perplexity and retrieval accuracy, and we profile the GPU kernels to explain the results.'),
    ('pc', 'The contributions are:'),
    ('p', '(1) A paired, repeated measurement of KV-cache compression energy on an edge device: 17 experiment sets, '
          '1K to 64K tokens, two power modes, up to eight concurrent sessions, with 95% confidence intervals and '
          'all raw data public {cite:repo}.'),
    ('p', '(2) The finding that in llama.cpp a quantized cache always costs energy on this device (+3% at 1K to '
          '+18% per request at 16K), and the mechanism: decode attention moves from a tensor-core kernel to a '
          'CUDA-core kernel that unpacks the cache and re-reads it for every query head.'),
    ('p', '(3) Where compression does pay: memory capacity. Only a quantized cache lets a 3B model reach 32K or '
          '64K tokens, or a 1B model serve eight 16K-token sessions, on 8 GB.'),
    ('p', '(4) Accuracy of every method on one retrieval task: 4-bit keys break the Qwen2.5 family; keeping keys '
          'at 8 or 16 bits avoids it at no more energy than a uniform 8-bit cache; eviction loses up to 95% of the '
          'correct answers; perplexity '
          'understates retrieval loss.'),
    ('p', '(5) Eviction and quantization in one engine (PyTorch): eviction saves 3–11% of the energy per request at '
          '8K–16K tokens, quantized caches cost 28–79% more, and an uncompressed cache in llama.cpp is cheaper than '
          'either; plus a comparison with the same settings on a data-center T4 GPU.'),
    ('p', '(6) A benchmarking pitfall: calling Hugging Face generate() with pad_token_id = 0 and no attention mask '
          'silently masks every occurrence of token 0 ("!" in both tokenizers) and moves attention to a slower path, '
          'which in our first runs looked like a 40–47% eviction saving.'),

    ('h', '2 Related Work', 1),
    ('p', '**KV-cache eviction.** H2O keeps "heavy hitter" tokens that have received the most attention so far '
          '{cite:h2o}. StreamingLLM observed that models place a large share of attention on the first tokens '
          '(attention sinks) and keeps those plus a window of recent tokens {cite:streamingllm}. SnapKV scores the '
          'prompt tokens by the attention they receive from an observation window at the end of the prompt, '
          'usually where the question is, and keeps the top tokens per head {cite:snapkv}. kvpress implements these '
          'and other methods as "presses" over Hugging Face models {cite:kvpress}. These papers report memory, '
          'speed and accuracy on server GPUs; none reports energy.'),
    ('p', '**KV-cache quantization.** KIVI quantizes keys per channel and values per token to 2 bits and keeps the '
          'most recent tokens in full precision {cite:kivi}; KVQuant adds per-channel and pre-RoPE key quantization '
          'and outlier handling {cite:kvquant}. Both observe that keys have outlier channels that make them harder '
          'to quantize than values, which our perplexity results confirm for the Qwen2.5 family. Serving systems '
          'such as QServe co-design 4-bit KV caches with attention kernels, because unpacking low-bit values inside '
          'attention can make decode compute-bound {cite:qserve}. Grouped-query attention (GQA) {cite:gqa} shrinks '
          'the cache architecturally; it also determines how much an attention kernel can share between query heads, '
          'which turns out to matter here.'),
    ('p', '**Kernels and engines.** FlashAttention fuses attention into one kernel that works on on-chip tiles '
          '{cite:flashattn}. llama.cpp implements flash attention with a tensor-core kernel for f16 caches and a '
          'vector kernel for quantized ones {cite:llamacpp}; PagedAttention manages the cache in server engines '
          '{cite:vllm}. Our results show that which kernel an engine selects for a cache format can matter more for '
          'energy than the size of the cache.'),
    ('p', '**Energy of LLM inference.** Samsi et al. measured the energy of LLM inference on data-center GPUs '
          '{cite:words2watts}. The T4 study this paper extends measured StreamingLLM and SnapKV through kvpress on '
          'a cloud Tesla T4 and found small, method- and model-dependent energy differences {cite:thesis}. Other '
          'measurements of KV-cache compression energy also come from data-center GPUs and disagree in sign. On a '
          'GH200, Hugging Face quantized caches (quanto, HQQ) raised GPU energy by 22–287% over an unquantized '
          'cache for 6.7B–70B models at batch size 1 {cite:vallecillos}, while in vLLM an FP8 cache lowered energy '
          'per token by a median 2.7% on A100, H100 and H200 GPUs {cite:powertrace}. At the edge, Kelle co-designs '
          'eviction with eDRAM and reports 4.5× energy savings, but for a simulated accelerator {cite:kelle}; '
          'mzCache manages the KV cache of on-device LLMs on phones and estimates energy from the battery sensor, '
          'but compares memory managers, not a compressed cache with an uncompressed one {cite:mzcache}. We are '
          'not aware of a measurement of the energy of existing KV-cache compression methods on a real edge device '
          'with a shared-memory GPU, which is the gap this paper addresses.'),
    ('p', '**Long-context evaluation.** Needle-in-a-haystack tests hide one fact in a long text and ask for it '
          '{cite:needle}; RULER adds distractor facts and other tasks to make retrieval harder {cite:ruler}. We '
          'use a needle test with three distractors so that a cache that lost part of the prompt cannot answer by '
          'returning the only code it still has.'),

    ('h', '3 System and Compression Methods', 1),
    ('h', '3.1 Platform', 2),
    ('p', 'All measurements run on one NVIDIA Jetson Orin Nano Developer Kit (8 GB): six Arm Cortex-A78AE cores and '
          'an Ampere GPU (compute capability 8.7) with 1,024 CUDA cores and 32 tensor cores, sharing 8 GB of LPDDR5 '
          '(7,850 MB visible to Linux, no swap). The board runs Ubuntu 24.04 with Jetson Linux R39.2 and CUDA '
          '13.2. Two power modes exist on this image: 15W (GPU up to 625 MHz, the default and the mode of every '
          'set) and 7W (408 MHz, four CPU cores), which we use for the main grid as a second operating point. The '
          'onboard TI INA3221 sensor reports voltage and current of three rails: VDD_IN (the whole module), '
          'VDD_CPU_GPU_CV (CPU, GPU and vision accelerators) and VDD_SOC (the rest of the chip, including the '
          'memory controller).'),
    ('h', '3.2 Engines and models', 2),
    ('p', '**llama.cpp** (commit 7adf160) is built twice for the GPU: a main build, and a second build with '
          'GGML_CUDA_FA_ALL_QUANTS=ON, which adds GPU flash-attention kernels for mixed key/value formats. Energy '
          'runs use llama-server with every layer on the GPU and flash attention on; llama-bench measures prompt '
          'processing and generation separately; llama-perplexity measures accuracy. **PyTorch** 2.15 (CUDA 13.2 '
          'build) {cite:pytorch} with transformers 4.57 runs the same models in fp16 with scaled-dot-product '
          'attention, kvpress 0.5.3 for eviction, and optimum-quanto 0.2.7 {cite:quanto} and HQQ 0.2.8 {cite:hqq} '
          'for quantized caches.'),
    ('tab', 'tab:models', 'Models. KV cache per token at f16 = layers × KV heads × head size × 2 × 2 bytes.',
     [['Model', 'Layers', 'Q / KV heads', 'KV/token', 'Weights'],
      ['Qwen2.5-0.5B {cite:qwen25}', '24', '14 / 2', '12.3 KB', 'Q8_0, Q4_K_M, fp16'],
      ['Qwen2.5-1.5B', '28', '12 / 2', '28.7 KB', 'Q4_K_M'],
      ['Qwen2.5-3B', '36', '16 / 2', '36.9 KB', 'Q4_K_M'],
      ['Llama-3.2-1B {cite:llama3}', '16', '32 / 8', '32.8 KB', 'Q8_0, Q4_K_M, fp16'],
      ['Llama-3.2-3B', '28', '24 / 8', '114.7 KB', 'Q4_K_M'],
      ['Gemma-3-1B {cite:gemma3}', '26 (4 global)', '4 / 1', '4.1 KB^a^', 'Q4_K_M']],
     [1.05, 0.5, 0.55, 0.5, 0.7],
     {'note': '^a^ In the 4 full-context layers; the other 22 layers keep a sliding window of 512 tokens.'}),
    ('p', 'Qwen2.5-0.5B and Llama-3.2-1B, the models of the T4 study, are used in every set. The four others '
          'extend the comparison to model size within a family and to a third family; Gemma-3 keeps the full '
          'context in only 4 of its 26 layers. All six use the instruction-tuned versions.'),
    ('h', '3.3 Compression methods', 2),
    ('p', '**llama.cpp cache formats.** f16 stores two bytes per value. q8_0 stores blocks of 32 values as 8-bit '
          'integers with one f16 scale (1.0625 bytes per value, 53% of f16); q4_0 stores 4-bit integers with one '
          'scale (0.5625 bytes, 28%); q4_1 adds a per-block minimum (0.625 bytes, 31%). Keys and values can use '
          'different formats; we call 8-bit keys with 4-bit values "q8_0/q4_0" and f16 keys with 4-bit values '
          '"f16/q4_0".'),
    ('p', '**Eviction (kvpress).** kvpress hooks every attention layer. At the end of prefill it scores every '
          'cached token per KV head, keeps the best (1 − *r*) × *N* tokens and deletes the rest, once per request; '
          'the generated tokens are added in full. StreamingLLM keeps 4 sink tokens and the most recent tokens. '
          'SnapKV averages the attention of the last 64 prompt tokens over each earlier token, smooths the scores '
          'over 5 neighbours, and keeps the top tokens per KV head. We use compression ratios *r* = 0.5 and 0.75. '
          'In the energy runs the question is part of the compressed prompt ("question inside"); the accuracy runs '
          'also compress the document alone and add the question afterwards ("question after"), as when one '
          'cached document serves several questions.'),
    ('p', '**Quantized cache in PyTorch.** transformers\' QuantizedCache follows KIVI {cite:kivi}: the newest 128 '
          'tokens stay in fp16, older ones are quantized in groups of 64 with a scale and a zero point, by quanto '
          '(4 or 2 bits) or HQQ (8 or 4 bits here). Every decode step dequantizes the stored cache to fp16 and then '
          'runs ordinary attention. On the T4 (compute capability 7.5) quanto\'s kernels did not build; on the Orin '
          'they do, which lets us compare quantization and eviction in one engine.'),

    ('h', '4 Methodology', 1),
    ('h', '4.1 Energy measurement', 2),
    ('p', 'A background thread reads the three INA3221 rails from sysfs every 50 ms (about 18 samples per second), '
          'with the GPU load counter, CPU busy time and free memory. Energy over a window is the trapezoid integral '
          'of power. *Gross* energy is what the module drew; *net* energy subtracts the idle power, measured for '
          '10 s before and after every experiment set (the lower reading is used), times the duration. We report '
          'net energy unless stated; gross energy leads to the same conclusions except where run times differ a lot '
          '(the 7W mode).'),
    ('p', '**Metrics.** (i) *Energy per request*: from sending a request to receiving the reply, prompt plus a '
          'fixed number of generated tokens (128 unless stated). (ii) *Decode energy per token*: energy after the '
          'prompt has been processed divided by the generated tokens. In llama-server runs the split uses the '
          'server\'s own prompt time; llama-bench times prompt processing (pp) and generation (tg) as separate '
          'tests, which needs no split; in PyTorch a stopping-criterion hook timestamps the first generated token. '
          '(iii) *Speed*: prefill and decode tokens per second. (iv) *Decode power*: mean VDD_IN power in the '
          'decode window, and energy per rail. (v) *Memory*: cache size and the largest context that fits. '
          '(vi) *Perplexity* on WikiText-2 {cite:wikitext}. (vii) *Retrieval accuracy* (Section 4.3). '
          '(viii) *Kernel time*: Nsight Systems kernel summaries and Nsight Compute counters, and torch.profiler '
          'for PyTorch.'),
    ('h', '4.2 Protocol and statistics', 2),
    ('p', 'Prompts are the first *N* tokens of a public-domain novel (*Pride and Prejudice*), tokenized by each '
          'model\'s own tokenizer and sent as token IDs, so every format sees exactly the same tokens. Decoding is '
          'greedy; replies have a fixed length (end-of-sequence is ignored), and prompt caching between requests is '
          'off; PyTorch runs pass the end-of-sequence token as padding (Section 5.11). Each round runs every configuration of a set once in an order shuffled anew per round. Before '
          'every set the desktop\'s software updater is closed (it kept one CPU core busy after boot). Every '
          'compressed configuration is compared with the uncompressed one *from the same round*: the change is the '
          'mean of the per-round ratios with a Student-t 95% confidence interval; a change whose interval contains '
          'zero is marked n.s. GPU clocks stayed at the mode maximum after every run (44–54 °C at 15W), so no run '
          'was throttled.'),
    ('h', '4.3 Accuracy tests', 2),
    ('p', 'Perplexity cannot score eviction, because kvpress compresses only once, after the prompt, so a '
          'perplexity run never decodes from the compressed cache. We therefore use both perplexity (WikiText-2, '
          'llama-perplexity, 2K and 8K windows) and a retrieval test that both engines run on the same token IDs. '
          'The retrieval prompt hides "The secret code for <city> is <7 digits>." at a depth of 10, 30, 50, 70 or '
          '90% of a passage of the novel, plus three distractor sentences for other cities at random depths '
          '{cite:needle,ruler}, and asks for the first city\'s code. An answer is correct if the seven digits appear '
          'in the first 16 greedy tokens. Six prompts per depth give 30 prompts per cell; we report accuracy with '
          'Wilson 95% intervals.'),
    ('h', '4.4 Experiment sets', 2),
    ('p', 'Table {ref:tab:sets} lists the sets. Sets A and B run at 15W and 7W, all others at 15W. Set letters are '
          'used throughout the paper and match the data files {cite:repo}.'),
    ('tab', 'tab:sets', 'Experiment sets. Rounds are paired repetitions of every configuration.',
     [['Set', 'What is measured', 'Models', 'Configurations', 'Contexts', 'Rounds'],
      ['A', 'Main grid, 128-token replies', 'Llama-1B, Qwen-0.5B (Q8_0)', 'f16 (FA on / off), q8_0, q4_0',
       '1K–16K', '5'],
      ['B', 'Prefill and decode separately (llama-bench)', 'same', 'same', 'pp and tg at 1K, 16K', '5'],
      ['C', 'Perplexity, keys and values split', 'same', 'f16, q8_0, q4_0, q4_1, mixed', '2K, 8K', '1'],
      ['D', 'Concurrency: 1, 2, 4, 8 sessions', 'same', 'f16, q8_0, q4_0', '4K, 16K each', '3'],
      ['E', 'Main grid with 4-bit weights', 'same (Q4_K_M)', 'f16, q8_0, q4_0', '1K–16K', '5'],
      ['F', 'Kernel profile (nsys) and counters (ncu)', 'same', 'f16 (FA on / off), q8_0, q4_0', 'depth 16K', '1'],
      ['G', 'Largest context that fits', 'Llama-1B, Llama-3B', 'f16, q8_0, q4_0', '16K–64K', '2'],
      ['H', 'Model size and family', 'six models (Q4_K_M)', 'f16, q8_0, q4_0', '1K–16K', '5'],
      ['I', 'Longer replies (512 tokens)', 'six models', 'f16, q8_0, q4_0', '4K, 16K', '3'],
      ['J', 'Perplexity of the six models', 'six models', 'f16, q8_0, q4_0, mixed', '2K, 8K', '1'],
      ['K', 'Eviction energy (PyTorch)^c^', 'Qwen-0.5B, Llama-1B (fp16)', 'none, StreamingLLM, SnapKV; 50/75%',
       '1K–16K', '5'],
      ['L', 'Mixed key/value formats', 'Llama-1B, Qwen-0.5B, Qwen-1.5B', 'f16, q8_0, q4_0, q8_0/q4_0, f16/q4_0',
       '1K–16K', '5'],
      ['M', 'Retrieval accuracy', 'Qwen-0.5B, Llama-1B', 'set L formats; eviction; quantized caches',
       '1K, 4K, 8K', '30 prompts'],
      ['N', 'Quantized cache energy (PyTorch)^c^', 'Qwen-0.5B, Llama-1B (fp16)', 'none, quanto 4/2, HQQ 8/4, SnapKV',
       '1K–8K', '3'],
      ['O', 'PyTorch decode step by code path', 'same', 'none', '2K–8K', '64 steps'],
      ['P', 'Eviction at the T4 study\'s settings^c^', 'same', 'none, StreamingLLM, SnapKV; 50%', '2K, 8K', '5'],
      ['Q', 'Perplexity of mixed formats', 'Qwen-1.5B (Q4_K_M)', 'f16, q8_0/q4_0, f16/q4_0', '8K', '1']],
     [0.35, 1.9, 1.55, 1.85, 0.85, 0.5],
     {'wide': True, 'note': '^c^ Cells with prompts of 8K tokens or more were rerun on 17 November 2026 with a '
                            'corrected padding token (Section 5.11).'}),

    ('h', '5 Results', 1),
    ('h', '5.1 A quantized cache costs energy at every length', 2),
    ('p', 'Table {ref:tab:main} gives the main result (set A). Every quantized cell uses more energy per request '
          'than f16 with flash attention, and every change is significant. The penalty grows with the context: '
          'about +3 to +5% at 1K and +13 to +18% at 16K. Per generated token it is two to three times larger, '
          '+29 to +58% at 16K, because prefill, which the cache format barely affects, dilutes the per-request '
          'number. The f16 baseline is the strong one: with flash attention off, f16 costs 3.3 to 3.7 times as '
          'much per request at 16K (+228% Llama, +270% Qwen).'),
    ('tab', 'tab:main', 'Set A, 15W: change against f16, paired, 95% CI over 5 rounds (net energy).',
     [['', 'Llama-1B 8-bit', 'Llama-1B 4-bit', 'Qwen-0.5B 8-bit', 'Qwen-0.5B 4-bit'],
      ['Request, 1K', '+4.3 ± 2.8%', '+2.6 ± 1.1%', '+4.6 ± 1.3%', '+4.4 ± 1.0%'],
      ['Request, 4K', '+6.7 ± 0.7%', '+5.9 ± 1.4%', '+10.9 ± 1.3%', '+10.3 ± 0.9%'],
      ['Request, 8K', '+12.0 ± 3.3%', '+9.3 ± 0.3%', '+16.8 ± 0.7%', '+13.5 ± 0.4%'],
      ['Request, 16K', '+15.9 ± 3.0%', '+12.6 ± 0.5%', '+18.0 ± 1.0%', '+16.0 ± 0.9%'],
      ['Per token, 16K', '+35.9 ± 2.5%', '+28.6 ± 0.8%', '+58.0 ± 3.3%', '+50.4 ± 3.7%'],
      ['Decode speed, 16K', f'{M2}19.9%', f'{M2}22.2%', f'{M2}33.3%', f'{M2}36.2%'],
      ['Cache size', f'{M2}47%', f'{M2}72%', f'{M2}47%', f'{M2}72%']],
     [0.85, 0.6, 0.6, 0.62, 0.62],
     {'note': 'f16 at 16K: Llama-1B 123.8 J per request (22.8 tok/s), Qwen-0.5B 66.0 J (51.0 tok/s).',
      'size': 7}),
    ('fig', 'fig:main', 'plots_v2/01_request_energy_change.png',
     'Set A: change in energy per request against f16, paired by round, 95% CI. Hollow markers: not separable '
     'from zero.'),
    ('p', 'Measuring prefill and decode separately (set B, llama-bench) confirms the split. Prompt processing at '
          '16K runs at nearly the same speed with every format (Llama 1,294 tok/s with f16, 1,250 and 1,242 with '
          '8-bit and 4-bit), while 128 tokens generated after a 16K-token cache cost 0.318 J per token with f16, '
          '0.434 J with 8-bit (+36.5%) and 0.412 J with 4-bit (+29.4%) on Llama, and +63.7% and +54.2% on Qwen. '
          'These agree with the server-run split within a few percent.'),
    ('h', '5.2 Why: the decode kernel changes', 2),
    ('p', 'Decode power barely changes with the cache format (Llama at 16K: 10.5, 11.2 and 10.5 W for f16, 8-bit '
          'and 4-bit); time grows, and energy with it. Nothing falls back to the CPU: in decode the CPU is 2–3% '
          'busy and the GPU 92–97% for every format. The extra decode energy appears on the CPU+GPU rail (+61 to '
          '+90% at 16K), while the SoC rail, which includes the memory controller, changes by 5 to 32%. If the '
          'cost were memory traffic, the SoC rail would carry it.'),
    ('p', 'The kernel profile (set F, Table {ref:tab:kern}) shows the cause. With an f16 cache, llama.cpp runs '
          'decode attention on a tensor-core (mma) flash-attention kernel that processes all query heads sharing '
          'a KV head together. With q8_0 or q4_0 it switches to a vector kernel on the CUDA cores that unpacks '
          'blocks as it reads them, one query row at a time. Per call the vector kernel takes 2.1–2.3 times as '
          'long on Llama and 3.5–3.9 times on Qwen. Hardware counters explain the difference: the f16 kernel is '
          'memory-bound (88% of peak memory throughput, 10% of peak compute on Llama), so a smaller cache could '
          'have made it faster; the vector kernels keep the SMs 64–74% busy and still run memory at 63–90% of '
          'peak, with 64–78% L1/L2 hit rates, because each query head of a group re-reads and re-unpacks the block '
          'it shares with the others. The compressed cache is smaller but is read 4 (Llama) or 7 (Qwen) times. '
          'Qwen, with 7 query heads per KV head, gains most from the shared tensor-core kernel and loses most '
          'when it is gone, which matches its larger energy penalty.'),
    ('tab', 'tab:kern', 'Set F: decode attention kernel at a 16K-token cache (Nsight Systems).',
     [['Model, cache', 'Kernel', 'µs per call', 'vs f16'],
      ['Llama-1B, f16', 'mma (tensor cores)', '570', '–'],
      ['Llama-1B, 8-bit', 'vec (CUDA cores)', '1,201', '+111%'],
      ['Llama-1B, 4-bit', 'vec (CUDA cores)', '1,309', '+130%'],
      ['Qwen-0.5B, f16', 'mma (tensor cores)', '151', '–'],
      ['Qwen-0.5B, 8-bit', 'vec (CUDA cores)', '532', '+253%'],
      ['Qwen-0.5B, 4-bit', 'vec (CUDA cores)', '589', '+291%']],
     [1.0, 1.05, 0.6, 0.55]),
    ('fig', 'fig:kern', 'plots_v2/09_kernels.png', 'Set F: decode attention kernel time per call at 16K.'),
    ('h', '5.3 The result holds across conditions', 2),
    ('p', '**Power mode.** At 7W all 48 quantized cells are again significantly more expensive than f16, and the '
          'penalty is larger (16K: Llama +21% / +18%, Qwen +28% / +21% for 8-bit / 4-bit), because the slower GPU '
          'makes the extra compute weigh more. The 7W mode itself costs energy: decode is 2.0–2.4 times slower '
          'while power falls only from about 10 W to 7 W, so a request uses 21–31% more net and 60–73% more '
          'gross energy than at 15W.'),
    ('p', '**Weight format.** With 4-bit (Q4_K_M) weights instead of 8-bit (set E) the penalties stay within a '
          'few points of set A (16K: +15.5%, +13.3%, +18.1%, +15.8%).'),
    ('p', '**Concurrency.** With 2, 4 and 8 sessions sent at once (set D), batching lowers the energy per '
          'generated token for every format, because the weights are read once per step for all sessions, but the '
          'quantized caches stay more expensive at every N (16K per session: Llama 8-bit +14.7% to +18.3%, Qwen '
          '8-bit +18.4% to +24.6%).'),
    ('p', '**Reply length.** With 512 instead of 128 generated tokens (set I) the per-request penalty grows 1.7–2.2 '
          'times (Qwen-0.5B 8-bit at 16K: +18% → +39%; Llama-1B: +16% → +29%), moving toward the per-token '
          'penalty as decode becomes most of the request. For chat use with long replies, the per-token numbers '
          'are the relevant ones.'),
    ('h', '5.4 Model size and family', 2),
    ('p', 'Table {ref:tab:size} shows the six models (set H). No quantized cell of the 144 comparisons is cheaper '
          'than f16 (143 significantly more expensive, one not separable from zero). Within a family the penalty '
          'shrinks with size: a larger model spends more of each decode step multiplying its weights, which costs '
          'the same with any cache. Gemma-3, which attends over the full context in only 4 of 26 layers, pays '
          'only +3–4%, but its cache is so small (90 MB at 16K) that compression also saves it almost nothing.'),
    ('tab', 'tab:size', 'Set H: change against f16 at 16K (net energy, 5 paired rounds).',
     [['Model', 'f16 cache', 'Request 8-bit', 'Request 4-bit', 'Token 8-bit', 'Token 4-bit'],
      ['Qwen2.5-0.5B', '203 MB', '+17.9%', '+16.0%', '+60.3%', '+52.1%'],
      ['Qwen2.5-1.5B', '474 MB', '+14.2%', '+12.5%', '+49.4%', '+43.8%'],
      ['Qwen2.5-3B', '608 MB', '+14.2%', '+13.6%', '+38.1%', '+36.4%'],
      ['Llama-3.2-1B', '541 MB', '+15.6%', '+13.5%', '+43.2%', '+35.2%'],
      ['Llama-3.2-3B', '1,894 MB', '+10.9%', '+9.7%', '+27.2%', '+21.8%'],
      ['Gemma-3-1B', '90 MB', '+3.6%', '+3.5%', '+3.5%', '+3.4%']],
     [0.8, 0.5, 0.5, 0.5, 0.5, 0.5], {'size': 7}),
    ('h', '5.5 Where compression pays: capacity', 2),
    ('p', 'Compression wins where f16 does not fit (set G, set D). Llama-3.2-3B with 4-bit weights fits a 16K f16 '
          'cache, but at 32K the f16 cache needs 3.8 GB and the server cannot start; 8-bit reaches 32K (893 J per '
          'request, 5.7 tok/s) and only 4-bit reaches 64K (2,471 J, 3.2 tok/s). Llama-1B with eight 16K-token '
          'sessions needs 4.3 GB of f16 cache, which does not fit next to the weights (starting the server anyway '
          'froze the board until it rebooted), while 8-bit (2.3 GB) and 4-bit (1.2 GB) run. Where f16 and a '
          'quantized cache both fit, f16 is cheaper: Llama-1B at 64K uses 761 J with f16 and 872 J with 8-bit. On '
          'this device, KV quantization buys context length and sessions and pays for them in energy and speed.'),
    ('h', '5.6 Accuracy: perplexity', 2),
    ('p', 'An 8-bit cache costs at most 1.5% perplexity on any model. A 4-bit cache splits by family (sets C, J): '
          'Llama and Gemma lose 1.5–5% (Llama-1B at 8K: 11.07 → 11.62), but every Qwen2.5 size breaks (8K: '
          '0.5B 12.8 → 179, 1.5B 8.7 → 6,425, 3B 7.6 → 35). Quantizing keys and values separately shows that '
          'the keys are the problem: Qwen-0.5B (Q8_0 weights, set C) at 8K with 4-bit values and f16 keys has perplexity 12.53 (f16: '
          '12.49), with 4-bit keys and f16 values 156.8. q4_1, which adds a per-block minimum, does not help '
          '(145.9), so an offset is not the cause; outlier channels in the keys are the likely one, as reported '
          'for other models {cite:kivi,kvquant}, made worse in Qwen2.5 by its key-projection bias and only two '
          'KV heads. 8-bit keys with 4-bit values stay within 1.6% of f16 on both models (Qwen 12.69, Llama '
          '11.10). On Qwen2.5-1.5B, the model whose 4-bit cache fails worst (set Q, 8K), f16 keys with 4-bit '
          'values give 8.71 and 8-bit keys with 4-bit values 8.72, against 8.67 for f16 and 6,425 for q4_0.'),
    ('h', '5.7 Mixed key/value formats', 2),
    ('p', 'Since only keys need the bits, 8-bit keys with 4-bit values (q8_0/q4_0) and f16 keys with 4-bit values '
          '(f16/q4_0) keep accuracy at most of the memory saving. Set L measures their energy, with every format '
          'rerun in the same all-quants build (Table {ref:tab:mix}). They do not escape the penalty. q8_0/q4_0 '
          'costs about what 8-bit keys and values cost; f16/q4_0 is the cheapest quantized format on every model '
          'and context but still significantly more expensive than f16 (+10 to +15% at 16K) while saving only '
          '36% of the cache. Any quantized half sends decode to the vector kernel: decode attention takes 564 µs '
          'per call with f16, 1,096 µs with f16/q4_0, 1,210 µs with q8_0 and 1,306 µs with q8_0/q4_0 (Llama-1B, '
          '16K).'),
    ('tab', 'tab:mix', 'Set L: energy per request against f16 at 16K, same build, 5 paired rounds.',
     [['Model', 'q8_0', 'q4_0', 'q8_0/q4_0', 'f16/q4_0'],
      ['Llama-1B', '+14.8%', '+12.8%', '+13.4%', '+10.1%'],
      ['Qwen-0.5B', '+18.4%', '+16.8%', '+17.8%', '+14.8%'],
      ['Qwen-1.5B', '+14.2%', '+12.7%', '+14.1%', '+10.7%'],
      ['Cache vs f16', '53%', '28%', '41%', '64%']],
     [0.8, 0.55, 0.55, 0.65, 0.65]),
    ('h', '5.8 Eviction energy in PyTorch', 2),
    ('p', 'llama.cpp has no eviction methods, so set K runs StreamingLLM and SnapKV through kvpress in PyTorch and '
          'compares them with the same models in PyTorch without eviction (Table {ref:tab:evict}). Llama-1B stops at '
          '8K: in fp16 PyTorch a 16K prompt does not fit next to the model. The cells at 8K and 16K come from the '
          'corrected rerun described in Section 5.11. Eviction saves energy, but little: at 1K it costs up to 2% '
          '(SnapKV\'s scoring raises prefill energy by about a third), at 4K it saves 1–6%, and at 8K–16K 3–11%. '
          'Removing 75% saves about 3 points more than removing 50%, and StreamingLLM saves 1–3 points more than '
          'SnapKV, which spends extra prefill work on scoring.'),
    ('tab', 'tab:evict', 'Set K: energy per request with eviction against no eviction (PyTorch, net, paired, 5 '
            'rounds). ^c^ = corrected rerun (Section 5.11).',
     [['Model, context', 'None: J', 'SLLM 50%', 'SLLM 75%', 'SnapKV 50%', 'SnapKV 75%'],
      ['Qwen-0.5B, 1K', '32', '+1.4%', '+1.3%', '+1.7%', '+1.7%'],
      ['Qwen-0.5B, 4K', '39', f'{M2}2.8%', f'{M2}4.5%', f'{M2}1.1%', f'{M2}2.7%'],
      ['Qwen-0.5B, 8K^c^', '55', f'{M2}4.8%', f'{M2}7.7%', f'{M2}2.8%', f'{M2}5.6%'],
      ['Qwen-0.5B, 16K^c^', '86', f'{M2}8.3%', f'{M2}11.1%', f'{M2}5.6%', f'{M2}8.6%'],
      ['Llama-1B, 1K', '55', '+0.4%', f'{M2}0.4%', '+0.9%', '+2.2% n.s.'],
      ['Llama-1B, 4K', '71', f'{M2}3.3%', f'{M2}6.1%', f'{M2}2.4%', f'{M2}4.9%'],
      ['Llama-1B, 8K^c^', '101', f'{M2}6.7%', f'{M2}10.2%', f'{M2}4.9%', f'{M2}8.0%']],
     [0.85, 0.42, 0.48, 0.5, 0.52, 0.6], {'size': 7}),
    ('p', 'Decode in PyTorch is launch-bound on this device: each generated token costs several hundred small GPU '
          'operations issued from Python, so decode runs at 13–14 tok/s (Qwen) and 15–18 tok/s (Llama) whatever the '
          'cache size, up to 16K tokens. Evicting half or three quarters of the cache therefore barely speeds decode '
          '(Qwen at 16K: 13.7 tok/s without eviction, 13.1–13.2 with) and reduces decode energy per token by only 4 '
          'to 15%. The engine matters more than eviction: Qwen-0.5B at 16K uses 86 J per request in PyTorch (fp16 '
          'weights) and 66 J in llama.cpp with an uncompressed f16 cache (8-bit weights); even with 75% evicted, '
          'PyTorch uses 77 J.'),
    ('h', '5.9 Retrieval accuracy of every method', 2),
    ('p', 'Table {ref:tab:needle} gives the retrieval test (set M) for all methods on the same prompts; Figure '
          '{ref:fig:needle} adds the intervals. With 30 prompts a cell has a 95% interval of roughly ±15 points '
          '(30/30 correct: 88.6–100%), so only large differences are meaningful. Four results stand out.'),
    ('tab', 'tab:needle', 'Set M: correct answers out of 30 (needle + 3 distractors). Left: Qwen2.5-0.5B, right: '
            'Llama-3.2-1B, at 1K / 4K / 8K-token prompts.',
     [['Engine, configuration', 'Qwen 1K', '4K', '8K', 'Llama 1K', '4K', '8K'],
      ['llama.cpp f16', '30', '24', '22', '25', '26', '27'],
      ['llama.cpp q8_0', '30', '23', '19', '25', '26', '26'],
      ['llama.cpp q4_0', '2', '1', '0', '26', '21', '16'],
      ['llama.cpp q8_0 K / q4_0 V', '30', '24', '21', '26', '25', '27'],
      ['llama.cpp f16 K / q4_0 V', '30', '24', '22', '25', '26', '26'],
      ['PyTorch, no compression', '30', '22', '22', '27', '25', '26'],
      ['PyTorch HQQ 8-bit', '30', '22', '22', '27', '25', '26'],
      ['PyTorch HQQ 4-bit', '8', '3', '2', '26', '25', '25'],
      ['PyTorch quanto 4-bit', '5', '4', '9', '27', '24', '26'],
      ['PyTorch quanto 2-bit', '0', '0', '0', '1', '0', '1'],
      ['StreamingLLM 50%, question inside', '18', '17', '15', '17', '15', '16'],
      ['StreamingLLM 75%, question inside', '6', '6', '5', '7', '5', '6'],
      ['SnapKV 50%, question inside', '8', '8', '8', '23', '24', '26'],
      ['SnapKV 75%, question inside', '8', '3', '4', '16', '26', '25'],
      ['StreamingLLM 50%, question after', '16', '13', '14', '15', '13', '15'],
      ['StreamingLLM 75%, question after', '6', '5', '6', '8', '4', '6'],
      ['SnapKV 50%, question after', '12', '1', '2', '7', '6', '4'],
      ['SnapKV 75%, question after', '9', '1', '1', '7', '4', '1']],
     [2.1, 0.6, 0.45, 0.45, 0.65, 0.45, 0.45], {'wide': True}),
    ('p', '**(1) 4-bit keys fail on Qwen in both engines.** llama.cpp q4_0 answers 0–2 of 30 on Qwen, and '
          'PyTorch\'s 4-bit caches (HQQ, quanto), which use a zero point and groups of 64, fail too (2–9). Keeping '
          'keys at 8 or 16 bits restores f16 accuracy (q8_0/q4_0 and f16/q4_0 within one answer of f16). 8-bit '
          'caches match f16 in both engines. 2-bit fails on both models.'),
    ('p', '**(2) Perplexity understates retrieval loss.** Llama-1B\'s 4-bit cache costs only 5% perplexity at 8K, '
          'but retrieval falls from 27 to 16 of 30 at 8K (and from 26 to 21 at 4K), outside the intervals of f16.'),
    ('p', '**(3) Eviction loses most of the retrieval accuracy.** StreamingLLM keeps the start and the end of the '
          'prompt, so it answers only when the needle is near the end: about half the prompts with 50% removed, '
          'one in five with 75%. SnapKV with the question inside the compressed prompt keeps Llama\'s accuracy '
          '(23–26 of 30 at 50%), because its observation window contains the question. Without the question '
          '("question after", the multi-question use case) it drops to 1–7 of 30. On Qwen, SnapKV fails even with '
          'the question: Qwen\'s tokenizer splits numbers into single digits, and per-head selection keeps only '
          'some of the seven digit tokens, so many answers are truncated codes.'),
    ('p', '**(4) Distractors matter.** Without distractors this test was too easy (every configuration answered '
          'almost every prompt in a pilot); with three distractor codes, StreamingLLM gives a distractor\'s code in '
          'up to 67 of 180 answers, which shows that the cache kept the wrong part of the prompt.'),
    ('fig', 'fig:needle', 'plots_v2/14_accuracy.png',
     'Set M: retrieval accuracy with 95% Wilson intervals, 30 prompts per bar.', {'wide': True, 'width': 5.8}),
    ('h', '5.10 Quantization and eviction in one engine', 2),
    ('p', 'In PyTorch (set N, Table {ref:tab:pytq}), every quantized cache uses more energy per request than the '
          'uncompressed cache at every length, and the penalty grows with the context. Every decode step '
          'dequantizes the whole stored cache to fp16 before attention, which adds work proportional to the cache: '
          'Llama at 8K decodes at 15.3 tok/s uncompressed and 7.6–8.3 tok/s with any quantized cache. HQQ also '
          'makes prefill much more expensive (+54 to +107% at 8K). SnapKV 50% in the same runs saves 3–5% at 8K. '
          'In one engine, eviction is the method that saves a little energy and quantization the one that keeps '
          'accuracy (HQQ 8-bit matches f16 in Table {ref:tab:needle}); no method here does both.'),
    ('tab', 'tab:pytq', 'Set N: quantized caches and SnapKV in PyTorch, energy per request against the '
            'uncompressed cache (net, paired, 3 rounds; 8K from the corrected rerun).',
     [['Model, context', 'quanto 4', 'quanto 2', 'HQQ 8', 'HQQ 4', 'SnapKV 50%'],
      ['Qwen-0.5B, 1K', '+13.3%', '+13.7%', '+13.7%', '+24.5%', '+2.0%'],
      ['Qwen-0.5B, 4K', '+19.9%', '+19.7%', '+27.6%', '+40.2%', f'{M2}1.4% n.s.'],
      ['Qwen-0.5B, 8K', '+29.6%', '+28.0%', '+44.3%', '+61.0%', f'{M2}2.7%'],
      ['Llama-1B, 1K', '+11.1%', '+10.7%', '+14.5%', '+20.0%', '+1.6%'],
      ['Llama-1B, 4K', '+28.0%', '+26.6%', '+39.7%', '+56.7%', f'{M2}1.8%'],
      ['Llama-1B, 8K', '+39.6%', '+36.5%', '+55.6%', '+78.5%', f'{M2}4.8%']],
     [0.85, 0.45, 0.45, 0.45, 0.45, 0.62], {'size': 7}),
    ('h', '5.11 A benchmarking pitfall: the padding token', 2),
    ('p', 'Our first PyTorch runs showed decode slowing down in one step between 6K and 8K cached tokens (Qwen '
          '13.6 → 8.5 tok/s, Llama 16.1 → 6.0) and eviction saving 40–47% at 8K by staying below the step. A '
          'profile of single decode steps (set O) did not reproduce it: the model\'s forward pass, the same pass with '
          'an all-ones attention mask, and generate() with an explicit mask all ran at a constant 70–74 ms per step '
          'for Qwen from 2K to 8K tokens, and Llama rose smoothly from 54 to 65 ms. The difference was one argument. Our benchmark '
          'called generate() with pad_token_id = 0 and no attention mask; generate() then infers the mask as "every '
          'token that is not the padding token", and token 0 is "!" in both tokenizers. The novel\'s first "!" is '
          'token 8,062 (Qwen) and 7,884 (Llama), so every prompt of 8K tokens or more got a mask with holes, and '
          'attention left its fast mask-free path. Adding the single "!" token to the prompt reproduces the step '
          'exactly (Table {ref:tab:pad}).'),
    ('tab', 'tab:pad', 'Set O: ms per decode step of generate() with pad_token_id = 0 and no mask, just before '
            'and after the first "!" token, and with the fix (explicit mask).',
     [['Model', 'Prompt tokens', 'pad = 0, no mask', 'Explicit mask'],
      ['Qwen-0.5B', '7,680', '72.4', '72.3'],
      ['Qwen-0.5B', '8,061 (no "!")', '72.4', '–'],
      ['Qwen-0.5B', '8,063 (one "!")', '113.3', '–'],
      ['Qwen-0.5B', '8,192', '115.8', '72.3'],
      ['Llama-1B', '7,884 (no "!")', '64.7', '–'],
      ['Llama-1B', '7,886 (one "!")', '153.9', '–'],
      ['Llama-1B', '8,192', '159.9', '65.1']],
     [0.75, 0.95, 0.8, 0.75]),
    ('p', 'We reran every PyTorch energy cell with a prompt of 8K tokens or more (sets K, N and P) with the end-of-'
          'sequence token as padding, which generate() never infers as padding inside a prompt; shorter prompts '
          'contain no "!" and were unaffected, and the retrieval test (set M) always passed an explicit mask. All '
          'PyTorch numbers in this paper are from the corrected runs. The error is easy to make and invisible in '
          'the output text, and in an energy benchmark it changes the conclusion: with it, eviction looked like a '
          '40–47% saving; without it, eviction saves 3–11%.'),
    ('h', '5.12 The same eviction on a data-center GPU', 2),
    ('p', 'Set P repeats the T4 study\'s settings on the Jetson: 2K and 8K prompts, 50 generated tokens, batch 1, '
          '50% eviction {cite:thesis}. The two devices now agree that eviction changes energy per request by a few '
          'percent (Table {ref:tab:t4}). In absolute terms a request costs 3.5 to 14 times less energy on the Jetson: '
          'Qwen at 2K uses 80.7 J in 1.8 s on the T4 and 16.1 J in 4.1 s on the Jetson; at 8K, 428 J in 7.8 s '
          'against 30.7 J in 5.5 s; Llama at 2K, 98.2 J against 27.9 J. The T4 numbers are GPU-only (NVML) and the '
          'Jetson numbers cover the whole module, and the two runs used different harnesses (the T4 study\'s own '
          'and ours), so only the order of magnitude should be compared.'),
    ('tab', 'tab:t4', 'Set P: energy per request with 50% eviction against none, at the T4 study\'s settings.',
     [['Device, model, context', 'StreamingLLM', 'SnapKV'],
      ['T4, Qwen-0.5B, 2K', f'{M2}6.9%', f'{M2}7.7%'],
      ['T4, Qwen-0.5B, 8K', f'{M2}3.2%', f'{M2}2.2%'],
      ['T4, Llama-1B, 2K', '+0.4%', '+3.0%'],
      ['Jetson, Qwen-0.5B, 2K', f'{M2}0.9% n.s.', '+1.5% n.s.'],
      ['Jetson, Qwen-0.5B, 8K', f'{M2}5.1%', f'{M2}0.7% n.s.'],
      ['Jetson, Llama-1B, 2K', '+0.3%', '+2.0%'],
      ['Jetson, Llama-1B, 8K', f'{M2}4.1%', f'{M2}2.3%']],
     [1.5, 0.8, 0.8], {'note': 'Jetson: net VDD_IN energy, paired, 5 rounds (8K from the corrected rerun). '
                               'T4: GPU energy from the thesis data, 15–20 runs per cell.'}),

    ('h', '6 Discussion', 1),
    ('p', '**Smaller is not cheaper.** The intuition behind expecting energy savings from compression is that decode '
          'is memory-bound and a smaller cache means fewer bytes. On this device the f16 decode kernel is indeed '
          'memory-bound, but the engine reaches the compressed cache through a different, compute-bound kernel. '
          'The energy result is a property of the kernel path, not of the cache size. This is a software limit: a '
          'kernel that unpacks blocks into shared memory once per KV-head group and then feeds the tensor cores is '
          'possible, and server systems build such kernels {cite:qserve}, but llama.cpp has none for single-token '
          'decode on this GPU. Until it does, the energy cost of a quantized cache on edge GPUs follows from how '
          'many query heads share a KV head (the more, the larger the loss) and how much of each decode step is '
          'attention (the larger the model, the smaller the loss).'),
    ('p', '**What to use.** For energy, an f16 cache whenever it fits. When memory forces compression, f16 keys '
          'with 4-bit values are the cheapest accurate choice (64% of the f16 cache, +10 to +15% energy at 16K), '
          '8-bit keys with 4-bit values the most compact accurate one (41%), and 4-bit keys should not be used on '
          'the Qwen2.5 family at all. Model choice matters as much: Gemma-3\'s sliding-window layers make its cache '
          'small enough that the question hardly arises.'),
    ('p', '**Eviction saves little, and pays in accuracy.** Decode in PyTorch on this device is limited by the '
          'cost of launching operations, not by reading the cache, so removing half or three quarters of the cache '
          'saves only 3–11% of a request\'s energy at 8K–16K tokens, consistent with the few percent the T4 study '
          'measured. The price is retrieval accuracy: it falls by half or more with StreamingLLM, and almost entirely '
          'with SnapKV when the question is not known at compression time; only SnapKV with the question inside the '
          'compressed prompt keeps Llama\'s accuracy. Choosing the engine matters more: an uncompressed cache in '
          'llama.cpp used less energy than PyTorch with 75% of the cache evicted.'),
    ('p', '**Check the harness.** A single generation argument (Section 5.11) turned a 3–11% saving into an '
          'apparent 40–47% one, with a plausible-looking explanation (a "slow attention path above 6K tokens"). '
          'Energy benchmarks of Hugging Face models should pass an explicit attention mask, and any step change in '
          'speed with context length deserves a check of what changed in the input.'),
    ('p', '**Accuracy needs task tests.** Perplexity ranked the cache formats correctly but not by how much they '
          'hurt: a 5% perplexity increase coincided with losing about 40% of the retrieval answers at 8K. Studies '
          'that justify a cache format by perplexity alone may understate its cost.'),

    ('h', '7 Limitations', 1),
    ('p', 'The study uses one device and one llama.cpp version; engines with fused low-bit attention kernels '
          '(for example TensorRT-LLM or MLC) could change the sign of the quantization result, and the claim is '
          'about the stack most edge users run. Models are small (0.5B–3B); 7B models do not fit next to a long '
          'f16 cache in 8 GB. Replies are greedy and of fixed length. Energy is measured at the module input, '
          'which includes the fan and regulators but not the carrier board. Eviction and quantized caches in '
          'PyTorch are compared with PyTorch, not with llama.cpp, and Llama-1B could not be run at 16K in '
          'PyTorch. The retrieval test has 30 prompts per cell and one task type; it detects large differences '
          'only. The T4 comparison mixes a GPU-only and a whole-module power measurement and two harnesses. The '
          'corrected PyTorch reruns (Section 5.11) ran on a different day from the original sets; each rerun '
          'compares against its own uncompressed baseline from the same rounds.'),

    ('h', '8 Future Work', 1),
    ('p', 'A decode kernel for quantized caches that keeps the tensor cores would test the paper\'s explanation '
          'directly: if it removes the penalty, quantization becomes an energy tool on edge GPUs too. Other '
          'engines (MLC, TensorRT-LLM) and a Jetson with a larger power envelope would show how general the '
          'result is. Eviction could be implemented in llama.cpp to measure '
          'its energy in an engine whose decode is not launch-bound. Longer retrieval suites (RULER {cite:ruler}) and multi-turn '
          'workloads would sharpen the accuracy comparison, and combining f16/q4_0 with Gemma-style sliding-window '
          'models is a promising low-memory, low-energy configuration.'),

    ('h', '9 Conclusion', 1),
    ('p', 'On an NVIDIA Jetson Orin Nano, compressing the KV cache with llama.cpp\'s 8-bit or 4-bit formats never '
          'saved energy: it cost 3% at short contexts and up to 18% per request (58% per token) at 16K, across '
          'six models, both power modes, both weight formats, up to eight sessions and longer replies, because '
          'the compressed cache moves decode attention off the tensor cores. What compression buys on this device '
          'is room: longer contexts and more sessions than f16 can hold. Keys must keep at least 8 bits on the '
          'Qwen2.5 family, and mixed formats keep accuracy without saving energy. In PyTorch, eviction saved 3–11% '
          'at long prompts at a large cost in retrieval accuracy, and quantized caches cost 28–79% more. For '
          'energy at the edge, the best KV cache is an uncompressed one in an efficient engine, and compression is '
          'the tool for when it does not fit.'),
    ('refs',),
]
