# Generation Tasks

 The conversations are converted into 842 tasks. A task is a conversation turn containing all previous turns together with the last user question (e.g., the task created for turn $k$ includes all user and agent questions/responses for the first $k-1$ turns plus the user question for turn $k$). Our generation tasks measure performance under three retrieval settings.

| Setting  | Description | File | # Tasks
| ------------- | ------------- |  ------------- |  ------------- | 
| Reference  | Generation using reference passages | [reference.jsonl](reference.jsonl) |  842 | 
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages. (Restricted to tasks that have 2 contexts or less) | [reference+RAG.jsonl](reference+RAG.jsonl) | 436 |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages. | [RAG.jsonl](RAG.jsonl) | 842 |

Generation experiments can be run using any desired models e.g. available on HuggingFace.