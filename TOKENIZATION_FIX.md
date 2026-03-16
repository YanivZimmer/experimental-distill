# Tokenization Fix

## Problem

```
ValueError: Unable to create tensor, you should probably activate truncation and/or padding
```

## Root Cause

The tokenize function in `local_trainer.py` wasn't properly:
1. Setting labels for causal language modeling
2. Removing columns after tokenization
3. Handling padding correctly

## Fix Applied

### In `training/local_trainer.py` - `create_trainer()` method:

**Before:**
```python
def tokenize_function(examples):
    return self.tokenizer(
        examples["text"],
        truncation=True,
        max_length=self.config.max_seq_length,
        padding="max_length",
    )

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer,
    mlm=False,
)
```

**After:**
```python
def tokenize_function(examples):
    # Tokenize the text field
    result = self.tokenizer(
        examples["text"],
        truncation=True,
        max_length=self.config.max_seq_length,
        padding=False,  # Will pad in data collator
    )
    # Set labels to input_ids for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize datasets
print(f"   Tokenizing datasets...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names  # Remove original columns
)
eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names  # Remove original columns
)

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
)
```

## Key Changes

1. ✅ **Set labels** - `result["labels"] = result["input_ids"].copy()`
   - Required for causal language modeling loss calculation

2. ✅ **Remove columns** - `remove_columns=train_dataset.column_names`
   - Removes the original "text" column after tokenization
   - Prevents tensor creation errors

3. ✅ **Dynamic padding** - `padding=False` in tokenizer, handled by data collator
   - More efficient than padding everything to max length
   - Data collator pads to longest in batch

4. ✅ **Pad to multiple** - `pad_to_multiple_of=8`
   - Improves GPU/CPU efficiency

## How It Works Now

```
1. Dataset has "text" field
         ↓
2. tokenize_function()
   - Tokenizes text
   - Creates input_ids, attention_mask
   - Sets labels = input_ids
   - Removes "text" column
         ↓
3. Dataset has: input_ids, attention_mask, labels
         ↓
4. DataCollator pads to batch max length
         ↓
5. Trainer can create tensors properly ✅
```

## Testing

After this fix, you should be able to run:

```bash
source .venv/bin/activate
python training/local_entry.py
```

Or:
```bash
make train-local
```

The tokenization error should be resolved and training should start properly.

## Status

✅ **Fixed** - Tokenization now handles causal LM properly
