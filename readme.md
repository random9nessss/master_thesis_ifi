# Project Name

## Project Description
Repository for the master's thesis at the [University of Zurich](https://uzh.ch/en.html) at the chair of [Computational Linguistics](https://www.cl.uzh.ch/en.html). The thesis explores email synthetization in maritime chartering and subsequent development and assesement of text-mining models. [AttrPrompting](https://proceedings.neurips.cc/paper_files/paper/2023/file/ae9500c4f5607caf2eff033c67daa9d7-Paper-Datasets_and_Benchmarks.pdf) and [Base-Refine](https://arxiv.org/abs/2502.01697) are covered for email synthetization. For text-ming, [GLiNER](https://arxiv.org/pdf/2311.08526), Template Filling with [Qwen](https://huggingface.co/Gensyn/Qwen2.5-0.5B-Instruct) and TF-IDF are evaluated under differet pretraining/finetuning objectives.

### API Configuration
To interact with the various language model APIs and HuggingFace, you need to create an `ENV.txt` file in the root directory of the project with the following API keys:

```
OPENAI_KEY=your_openai_api_key_here
ANTHROPIC_KEY=your_anthropic_api_key_here
GOOGLE_KEY=your_google_api_key_here
MISTRAL_KEY=your_mistral_api_key_here
DEEPSEEK_KEY=your_deepseek_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## Project Organization

    ├── README.md          <- The top-level README for developers using this project
    ├── analytics          <- Core analysis modules for text evaluation
    │   ├── __init__.py
    │   ├── lexical_diversity.py               <- Lexical diversity metrics implementation
    │   ├── readability_score.py               <- Text readability scoring algorithms
    │   ├── run_diversity_analysis_ablation.py <- Ablation study for diversity analysis
    │   ├── run_diversity_analysis_csv.py      <- Diversity analysis script, creating result file
    │   ├── semantic_diversity.py              <- Semantic diversity analysis
    │   ├── sentiment_analysis.py              <- Sentiment analysis implementation
    │   ├── syntactic_diversity.py             <- Syntactic diversity metrics
    │   └── verbosity_analysis.py              <- Text verbosity analysis
    │
    ├── apiclients         <- API client implementations for various LLM providers
    │   ├── __init__.py
    │   ├── anthropic.py             <- Anthropic API client
    │   ├── deepseek.py              <- DeepSeek API client
    │   ├── google.py                <- Google API client
    │   ├── mistral.py               <- Mistral API client
    │   └── openai.py                <- OpenAI API client
    │
    ├── attributes         <- Attribute extraction and analysis modules
    │   ├── __init__.py
    │   ├── email_attribute_sampler.py  <- attribute sampler for email synthetization
    │   ├── few_shot_candidates.py      <- few shot candidate sampling
    │   ├── few_shot_prompt.py          <- few shot prompting script
    │   ├── news_attribute_sampler.py   <- attribute sampler for news synthetization
    │   └── zero_shot_prompt.py         <- zero shot prompting script
    │
    ├── attprompting       <- Prompt engineering modules for different APIs
    │   ├── __init__.py
    │   ├── attr_prompting_anthropic.py <- attribute prompting script for Anthropic API
    │   ├── attr_prompting_deepseek.py  <- attribute prompting script for DeepSeek API
    │   ├── attr_prompting_google.py    <- attribute prompting script for Google API
    │   ├── attr_prompting_mistral.py   <- attribute prompting script for Mistral API
    │   └── attr_prompting_openai.py    <- attribute prompting script for OpenAI API
    │
    ├── bare               <- Base model refinement and execution scripts
    │   ├── __init__.py
    │   ├── base.py     <- base script for BARE
    │   ├── refine.py     <- refinement script for BARE
    │   ├── refine_anthropic.py     <- refinement script for Anthropic API
    │   ├── refine_deepseek.py      <- refinement script for DeepSeek API
    │   ├── refine_google.py        <- refinement script for Google API
    │   ├── refine_mistral.py       <- refinement script for Mistral API
    │   ├── refine_openai.py        <- refinement script for OpenAI API
    │   ├── run_base_linux.py       <- refinmenet wrapper for Linux envirornment
    │   ├── run_baserefine.py       <- executable for baserefine
    │   └── run_refine.py           <- executable for refine step
    │
    ├── codeIE             <- Haversine distance based freight quote computation
    │   ├── codeie_vs_normal_ie.py      <- comparison of code-based and standard LLMs in IE
    │   ├── results                     <- result directory
    │
    ├── config             <- Configuration files and logging setup
    │   ├── __init__.py
    │   └── logger.py               <- logging script
    │
    ├── contrastive_finetuning <- Encoder finetuning experiments and implementations
    │   ├── contrastive_label_trainer       <- Contrastive learning (entity-level)
    │   │   ├── batchsampler.py            <- Batch sampling for contrastive learning
    │   │   ├── datapreparator.py          <- Data preparation utilities
    │   │   ├── entityextractor.py         <- Entity extraction from text
    │   │   ├── entitytypesimilaritymanager.py <- Entity type similarity management
    │   │   ├── logger.py                  <- Logging utilities for training
    │   │   ├── main.py                    <- Main execution script
    │   │   └── trainer.py                 <- Contrastive trainer
    │   │
    │   ├── eval                           <- Evaluation scripts and results
    │   │   ├── evaluation_results         <- Directory for storing evaluation outputs
    │   │   └── contrastive_evaluation.py  <- Contrastive model evaluation script (MTEB)
    │   │
    │   └── simcse                         <- Contrastive learning (sentence-level)
    │       └── contrastive_trainer.py     <- SimCSE trainer
    │
    ├── data               <- Data directory structure
    │   ├── email_datasets <- Email corpora for analysis
    │   │   ├── enron
    │   │   │   ├── enron_parsed.csv
    │   │   │   ├── enron_processed.csv
    │   │   │   ├── enron_processed_condensed.csv
    │   │   │   └── enron_processed_condensed_sampled_450.csv
    │   │   └── synthetic
    │   │       ├── attprompting      <- Attribute prompting generated emails
    │   │       │   ├── claude
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── deepseek
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gemini
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gpt-4-turbo
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   └── mistral
    │   │       │       └── aggregated
    │   │       │           └── aggregated.json
    │   │       ├── baserefine        <- BARE method generated emails
    │   │       │   ├── base
    │   │       │   │   ├── llama3b
    │   │       │   │   │   └── base_chains.json
    │   │       │   │   └── llama8b
    │   │       │   │       └── base_chains.json
    │   │       │   └── refine
    │   │       │       ├── llama3b
    │   │       │       │   ├── claude
    │   │       │       │   │   └── aggregated
    │   │       │       │   │       └── aggregated.json
    │   │       │       │   ├── deepseek
    │   │       │       │   │   └── aggregated
    │   │       │       │   │       └── aggregated.json
    │   │       │       │   ├── gemini
    │   │       │       │   │   └── aggregated
    │   │       │       │   │       └── aggregated.json
    │   │       │       │   ├── gpt-4-turbo
    │   │       │       │   │   └── aggregated
    │   │       │       │   │       └── aggregated.json
    │   │       │       │   └── mistral
    │   │       │       │       └── aggregated
    │   │       │       │           └── aggregated.json
    │   │       │       └── llama8b
    │   │       │           ├── claude
    │   │       │           │   └── aggregated
    │   │       │           │       └── aggregated.json
    │   │       │           ├── deepseek
    │   │       │           │   └── aggregated
    │   │       │           │       └── aggregated.json
    │   │       │           ├── gemini
    │   │       │           │   └── aggregated
    │   │       │           │       └── aggregated.json
    │   │       │           ├── gpt-4-turbo
    │   │       │           │   └── aggregated
    │   │       │           │       └── aggregated.json
    │   │       │           └── mistral
    │   │       │               └── aggregated
    │   │       │                   └── aggregated.json
    │   │       ├── fewshot           <- Few-shot prompting generated emails
    │   │       │   ├── claude
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── deepseek
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gemini
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gpt-4-turbo
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   └── mistral
    │   │       │       └── aggregated
    │   │       │           └── aggregated.json
    │   │       ├── iterativebaserefine <- Iterative BARE generated emails
    │   │       │   ├── claude
    │   │       │   │   └── aggregated.json
    │   │       │   └── mistral
    │   │       │       └── aggregated.json
    │   │       ├── zeroshot          <- Zero-shot prompting generated emails
    │   │       │   ├── claude
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── deepseek
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gemini
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   ├── gpt-4-turbo
    │   │       │   │   └── aggregated
    │   │       │   │       └── aggregated.json
    │   │       │   └── mistral
    │   │       │       └── aggregated
    │   │       │           └── aggregated.json
    │   │       └── train_email_synthetic.json
    │   │
    │   ├── gliner_finetuning <- Finetuning data for GLiNER
    │   │
    │   ├── maritime_template_data <- Template data gemeratopms for finetuning
    │   │   └── template_generated_maritime_ner_dataset.json
    │   │
    │   ├── newsarticles   <- News article collections
    │   │   ├── gcaptain
    │   │   │   └── gcaptain_articles.json
    │   │   ├── marinetraffic
    │   │   │   └── marinetraffic_articles.json
    │   │   ├── maritimeexecutive
    │   │   │   └── maritimeexecutive_articles.json
    │   │   ├── splash247
    │   │   │   └── splash247_articles.json
    │   │   ├── news_articles_combined.json
    │   │   └── news_articles_nuner_format.json
    │   │
    │   ├── nli            <- Natural Language Inference datasets
    │   │   └── nli_for_simcse.csv
    │   │
    │   ├── ood_ner_benchmark <- Out-of-domain NER benchmark datasets
    │   │   ├── ai.json
    │   │   ├── literature.json
    │   │   ├── movie.json
    │   │   ├── music.json
    │   │   ├── politics.json
    │   │   ├── restaurant.json
    │   │   └── science.json
    │   │
    │   ├── pilener        <- GLINER training dataset
    │   │   └── pilener_train.json
    │   │
    │   ├── port_data      <- Port and location data
    │   │   ├── geocoding
    │   │   │   ├── distance_matrix.xlsx
    │   │   │   └── port_data_geocoded.xlsx
    │   │   ├── shipping_ports_around_the_world
    │   │   │   └── port_data.csv
    │   │   └── unlcode
    │   │       ├── unlocode_checkpoint.csv
    │   │       ├── unlocode_complete_20250604_144152.csv
    │   │       └── unlocode_ports_only_20250604_144152.csv
    │   │
    │   ├── ships_data     <- Ship and vessel information
    │   │   ├── global_cargo_ships
    │   │   │   └── ships_data.csv
    │   │   └── imo
    │   │       └── imo_vessel_data_cleaned.csv
    │   │
    │   └── wikipedia      <- Wikipedia corpus and articles
    │       ├── chartering_corpus.json
    │       ├── maritime_articles.json
    │       ├── visited_urls.json
    │       ├── wikipedia_nlm_data.json
    │       └── wikipedia_nuner_format.json
    │
    ├── data_preprocessing  <- Data preprocessing and transformation scripts
    │   ├── enron_preprocessing.py    <- Enron dataset preprocessing
    │   ├── imo_preprocessing.py      <- IMO vessel data preprocessing
    │   ├── pilener_preprocessing.ipynb <- PileNER dataset preprocessing
    │   └── wikipedia_preprocessing.py  <- Wikipedia corpus preprocessing
    │
    ├── datagenerator  <- Data Generation
    │   ├── __init__.py
    │   ├── email_generator.py      <- email generation (AttrPrompting template)
    │   ├── news_generator.py       <- news articles generation (AttrPrompting)
    │
    ├── detectgpt  <- Authorship Detection
    │   ├── __init__.py
    │   ├── detect_gpt.py           <- detectgpt authorshipdetection
    │   ├── run_detect_gpt.py       <- execution script for detectgpt analysis
    │
    ├── enron  <- Human Baseline Corpus
    │   ├── __init__.py
    │   ├── enron_email_parser.py           <- email parser for the human-authored Enron corpus
    │
    ├── gliner_evaluation  <- In-domain Evaluation GLiNER
    │   ├── results
    │   ├── evaluation.py           <- evaluation script GLiNER in-domain
    │
    ├── gliner_finetuning  <- GLiNER finetuning scripts
    │   ├── finetune.py                      <- gliner regular finetuning
    │   ├── finetune_curriculum.py           <- gliner curriculum finetuning
    │   ├── finetune_membank.py              <- gliner memorybank contrastive finetuning
    │
    ├── gliner_training <- GLiNER training scripts
    │   ├── config.yaml              <- training configuration
    │   ├── data_loading.py          <- data loading utils functions
    │   ├── evaluation.py            <- evaluation script (during training)
    │   ├── gliner_trainer.py        <- gliner trainer script
    │   ├── utils.py                 <- utils script (incl. threshold scheduling, backend config, etc.)
    │
    ├── iterativebaserefine <- Iterative BARE variant for ablation study
    │   ├── results                 <- result directory
    │   ├── __init__.py
    │   ├── anthropic.py            <- iterative BARE using antrhopic refiner
    │   ├── mistral.py              <- iterative BARE using mistral refiner
    │
    ├── linuxvdi  <- Linux application for base generations in BARE
    │   ├── __init__.py
    │   ├── llm_local_app.py           <- local FastAPI instance
    │
    ├── llm_finetuning  <- Qwen finetuning and evaluation
    │   ├── eval_results
    │   ├── qwen_evaluation.py           <- evaluation of Qwen variants
    │   ├── qwen_finetuning.py           <- Qwen finetuning script
    │
    ├── lmstudio  <- LMStudio based local hosting of DeepSeek
    │   ├── __init__.py
    │   ├── deepseek_local.py           <- local DeepSeek hosting
    │
    ├── mlm_finetuning  <- MLM Finetuning of DeBERTa
    │   ├── __init__.py
    │   ├── mlm_trainer.py           <- MLM training script
    │
    ├── models             <- Trained and finetuned. model checkpoints
    │   ├── DeBERTa        <- DeBERTa encoder variants
    │   │   ├── bge                          <- Contrastive finetuned. (entity-level)
    │   │   ├── mlm                          <- MLM finetuned.
    │   │   ├── sbert                        <- Contrastive finetuned. (entity-level)
    │   │   └── simcse                       <- Contrastive finetuned. (sentence-level)
    │   │
    │   ├── GLiNER         <- GLiNER model variants and checkpoints
    │   │   ├── base                         <- Base encoder variants
    │   │   │   ├── bge                      <- BGE encoder base model
    │   │   │   ├── mlm                      <- MLM pre-trained encoder base model
    │   │   │   ├── sbert                    <- Sentence-BERT encoder base model
    │   │   │   └── simcse                   <- SimCSE encoder base model
    │   │   │
    │   │   ├── email                        <- Email-specific finetuned. models
    │   │   │   ├── gliner_base_email        <- Base GLiNER finetuned. on email data
    │   │   │   ├── gliner_bge_email         <- BGE-based GLiNER finetuned. on email data
    │   │   │   ├── gliner_mlm_email         <- MLM-based GLiNER finetuned. on email data
    │   │   │   ├── gliner_sbert_email       <- SBERT-based GLiNER finetuned. on email data
    │   │   │   └── gliner_simcse_email      <- SimCSE-based GLiNER finetuned. on email data
    │   │   │
    │   │   ├── finetuned                    <- Standard finetuned. models
    │   │   │   ├── normal_finetune_gliner_base    <- finetuned. base GLiNER (maritime corpus)
    │   │   │   ├── normal_finetune_gliner_bge     <- finetuned. BGE-based GLiNER (maritime corpus)
    │   │   │   ├── normal_finetune_gliner_mlm     <- finetuned. MLM-based GLiNER (maritime corpus)
    │   │   │   ├── normal_finetune_gliner_sbert   <- finetuned. SBERT-based GLiNER (maritime corpus)
    │   │   │   └── normal_finetune_gliner_simcse  <- finetuned. SimCSE-based GLiNER (maritime corpus)
    │   │   │
    │   │   └── membank                      <- Memory bank contrastive finetuned. models
    │   │       ├── gliner_membank_base      <- Memory bank finetuned. base GLiNER
    │   │       ├── gliner_membank_bge       <- Memory bank finetuned. BGE-based GLiNER
    │   │       ├── gliner_membank_mlm       <- Memory bank finetuned. MLM-based GLiNER
    │   │       ├── gliner_membank_sbert     <- Memory bank finetuned. SBERT-based GLiNER
    │   │       └── gliner_membank_simcse    <- Memory bank finetuned. SimCSE-based GLiNER
    │   │
    │   └── Qwen           <- Qwen language model variants
    │       ├── base                         <- Base Qwen model
    │       ├── qwen_emails                  <- PEFT Qwen variant (email data)
    │       └── qwen_no_emails               <- PEFT Qwen variant (maritime corpus)
    │
    ├── notebooks          <- Jupyter notebooks for exploration and analysis
    │   ├── ablation_bare_sentiment.ipynb           <- Iterative/Standard BARE ablation study (sentiment)
    │   ├── chartering_articles_annotation.ipynb    <- Maritime chartering article annotation
    │   ├── diversity_visualization.ipynb           <- Diversity metrics visualization
    │   ├── embedding_visualizer.ipynb              <- Embedding t-SNE visualization
    │   ├── qwen_gliner_comparison.ipynb            <- Comparison visualization Qwen vs. GLiNER
    │   ├── geocoding.ipynb                         <- Port and location geocoding
    │   ├── gliner_evaluation_visualizations.ipynb  <- GLiNER evaluation result visualizations
    │   ├── qwen_evaluation_vis.ipynb               <- Qwen evaluation visualizations
    │   ├── port_vessel_contrasting_dev.ipynb       <- Port-vessel contrastive visualization
    │   ├── sentiment_enron_corpus.ipynb            <- Sentiment analysis on Enron corpus
    │   ├── synthetic_emails_visualizations.ipynb   <- Synthetic email visualizations (diversity, verbosity, sentiment)
    │   ├── template_data_generation.ipynb          <- Template data generation for finetuning
    │   ├── unlocode_visualizer.ipynb               <- UN/LOCODE Port visualization
    │   ├── visualizer_detect_gpt.ipynb             <- DetectGPT visualization
    │   └── word_cloud_generation.ipynb             <- Word cloud visualizations
    │
    ├── output             <- Generated analysis visualizations
    │   ├── clustering                  <- Clustering analysis results
    │   ├── codele                      <- CodeIE analysis outputs
    │   ├── detectgpt                   <- DetectGPT analysis results
    │   ├── diversity                   <- Diversity analysis outputs
    │   ├── embeddings_tsne             <- t-SNE embedding visualizations
    │   ├── gliner_evaluation           <- GLiNER evaluation results
    │   ├── gliner_qwen_comparison      <- GLiNER vs Qwen comparison analysis
    │   ├── llama3b_sentiment           <- Llama 3B sentiment analysis results
    │   ├── llama8b_sentiment           <- Llama 8B sentiment analysis results
    │   ├── pilener_preprocessing       <- PileNER preprocessing outputs
    │   ├── port_maps                   <- Port location map visualizations
    │   ├── qwen_evaluation             <- Qwen evaluation results
    │   ├── sentiment                   <- Sentiment analysis outputs
    │   ├── sentiment_distribution      <- Sentiment distribution analysis
    │   ├── textualdiversity            <- Textual diversity metrics
    │   ├── verbosity                   <- Verbosity analysis results
    │   └── wordcloud                   <- Generated word clouds
    │
    ├── quotegenerator             <- Haversine distance based freight quote computation
    │   ├── __init__.py
    │   ├── quote_generator.py           <- freight quote computation engine
    │
    ├── tfidf        <- TFIDF Evaluation
    │   ├── results       <- result directory
    │   ├── tfidf.py      <- tfidf email evaluation script
    │
    ├── webscraping             <- Webscraping Scripts
    │   ├── __init__.py
    │   ├── gcaptain.py                     <- Scraper for gcaptain news outlet
    │   ├── imo_vessel_scraper.py           <- IMO Vessel data scraper
    │   ├── maritime_executive.py           <- Scraper for maritime executive news outlet
    │   ├── maritime_traffic.py             <- Scraper for maritime traffic news outlet
    │   ├── splash247.py                    <- Scraper for splash 247 news outlet
    │   ├── unlocode_scraper.py             <- UN/LOCODE data scraper
    │
    ├── wikipedia        <- Wikipedia Knowledge Graph Scraper
    │   ├── __init__.py
    │   ├── wikipedia_postprocessor.py       <- wikipedia data postprocessor
    │   ├── wikipedia_scraper.py             <- wikipedia scraper (snowball search on seed articles)
    │   ├── wikipedia_scraper_exec.py        <- executor scripy
    │
    ├── zeroshotprompting  <- Zero-shot prompting implementations
    │   ├── init.py
    │   ├── zero_shot_anthropic.py          <- Zero-shot prompting for Anthropic API
    │   ├── zero_shot_deepseek.py           <- Zero-shot prompting for DeepSeek API
    │   ├── zero_shot_google.py             <- Zero-shot prompting for Google API
    │   ├── zero_shot_mistral.py            <- Zero-shot prompting for Mistral API
    │   └── zero_shot_openai.py             <- Zero-shot prompting for OpenAI API
