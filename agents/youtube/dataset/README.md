# youtube dataset

Place JSONL splits here (contents are git-ignored):

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

Each line is a JSON object, either instruction-style:

```json
{"instruction": "...", "input": "", "output": "..."}
```

or chat-style:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
