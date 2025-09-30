# JSON Chunking and Humanization Documentation

## Overview

The `_read_json_file_chunks` function provides intelligent processing of JSON files by:

1. **Recursively analyzing JSON structure** to find optimal chunk boundaries
2. **Keeping chunks under 1KB** while preserving semantic meaning
3. **Humanizing JSON content** to make it more readable and searchable
4. **Maintaining context** through hierarchical path information

## Key Features

### 🎯 **Smart Chunking Strategy**

The function finds the **highest-level nodes** that fit within 1KB, ensuring:
- **Semantic coherence**: Related data stays together
- **Size compliance**: Each chunk is under 1KB
- **Context preservation**: Path information provides structure context

#### **Chunking Logic:**

```
JSON Object/Array
├── Try to fit entire key-value pair in 1KB
├── If too large → Recurse into the value
├── If primitive is too large → Split text intelligently
└── Maintain path context (e.g., "user.preferences.theme")
```

### 🤖 **JSON Humanization**

Converts technical JSON into natural language:

#### **Before (Raw JSON):**
```json
{
  "user_id": 12345,
  "is_active": true,
  "last_login": null,
  "account_balance": 150.75,
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```

#### **After (Humanized):**
```
User Id is 12345. Is Active is yes. Last Login is not specified. Account Balance is 150.75. Preferences contains: theme is dark, notifications is yes.
```

## Function Architecture

### Core Functions

#### 1. `_read_json_file_chunks(file_path)`
- **Entry point** for JSON file processing
- **Auto-detects** JSON files by extension
- **Yields** humanized text chunks with metadata

#### 2. `_extract_json_chunks(data, max_size, current_path)`
- **Recursive processor** that traverses JSON structure
- **Finds optimal boundaries** for 1KB chunks
- **Maintains path context** for nested structures

#### 3. `_humanize_json_object(obj, path)`
- **Converts objects** to natural language
- **Handles nested structures** with smart descriptions
- **Formats keys** (snake_case → Title Case)

#### 4. `_humanize_json_array_item(item, path, index)`
- **Processes array elements** with position context
- **Describes item types** and content summaries
- **Numbers items** for clarity (Item 1, Item 2, etc.)

#### 5. `_split_large_text(text, max_size)`
- **Splits oversized content** at sentence/word boundaries
- **Preserves meaning** by avoiding mid-sentence breaks
- **Handles edge cases** like very long words

## Humanization Rules

### 🔧 **Object Processing**

| JSON Type | Humanization Strategy | Example |
|-----------|----------------------|---------|
| **String** | Direct value with context | `"name": "John"` → `"Name is John"` |
| **Number** | Value with units/context | `"age": 30` → `"Age has value: 30"` |
| **Boolean** | Yes/No conversion | `"active": true` → `"Active is yes"` |
| **Null** | "Not specified" | `"phone": null` → `"Phone is not specified"` |
| **Object** | Nested key summary | `"address": {...}` → `"Address contains: street, city, zip"` |
| **Array** | Count and type description | `"items": [...]` → `"Items contains 3 objects with keys like name, price"` |

### 🎨 **Key Formatting**

- **snake_case** → **Title Case**: `user_name` → `User Name`
- **kebab-case** → **Title Case**: `last-login` → `Last Login`
- **Context preservation**: Full path maintained for nested access

### 📊 **Array Handling**

```json
{
  "products": [
    {"name": "Laptop", "price": 999.99},
    {"name": "Mouse", "price": 29.99},
    {"name": "Keyboard", "price": 79.99}
  ]
}
```

**Becomes:**
```
Products contains 3 items including objects. Item 1: Name is Laptop. Price is 999.99. Item 2: Name is Mouse. Price is 29.99. Item 3: Name is Keyboard. Price is 79.99.
```

## Usage Examples

### Example 1: E-commerce Data

**Input JSON:**
```json
{
  "order": {
    "order_id": "ORD-12345",
    "customer": {
      "name": "John Doe",
      "email": "john@example.com"
    },
    "items": [
      {"product": "Laptop", "quantity": 1, "price": 999.99},
      {"product": "Mouse", "quantity": 2, "price": 29.99}
    ],
    "total": 1059.97,
    "status": "shipped"
  }
}
```

**Generated Chunks:**
```
Chunk 1: "Order contains: Order Id is ORD-12345, Customer contains name, email, Items contains 2 objects, Total is 1059.97, Status is shipped."

Chunk 2: "Item 1: Product is Laptop. Quantity is 1. Price is 999.99."

Chunk 3: "Item 2: Product is Mouse. Quantity is 2. Price is 29.99."
```

### Example 2: User Profile

**Input JSON:**
```json
{
  "user_profile": {
    "personal_info": {
      "first_name": "Alice",
      "last_name": "Smith", 
      "age": 28,
      "email": "alice.smith@email.com"
    },
    "preferences": {
      "theme": "dark",
      "notifications": true,
      "language": "English"
    },
    "activity": {
      "last_login": "2025-01-15T10:30:00Z",
      "login_count": 42,
      "premium_member": true
    }
  }
}
```

**Generated Chunks:**
```
Chunk 1: "Personal Info contains: First Name is Alice, Last Name is Smith, Age is 28, Email is alice.smith@email.com."

Chunk 2: "Preferences contains: Theme is dark, Notifications is yes, Language is English."

Chunk 3: "Activity contains: Last Login is 2025-01-15T10:30:00Z, Login Count is 42, Premium Member is yes."
```

## Integration with Pipeline

### **Automatic Detection**

The main `_read_file_chunks` function automatically detects JSON files:

```python
def _read_file_chunks(self, file_path):
    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(filename)
    
    # Use specialized JSON processing for JSON files
    if ext.lower() == '.json':
        yield from self._read_json_file_chunks(file_path)
        return
    
    # Default text processing for other files
    # ... rest of function
```

### **Seamless Processing**

JSON files are processed automatically in the RAG pipeline:

1. **Detection**: `.json` extension triggers specialized handling
2. **Chunking**: Recursive analysis creates optimal 1KB chunks
3. **Humanization**: Technical JSON becomes searchable text
4. **Embedding**: Humanized chunks get embedded like regular text
5. **Storage**: Stored in Qdrant with appropriate metadata

## Benefits

### 🎯 **For Search & Retrieval**

- **Better matches**: Humanized text matches natural language queries
- **Context preservation**: Hierarchical structure maintains relationships
- **Semantic coherence**: Related data stays together in chunks

### 📊 **For Data Analysis**

- **Readable content**: Technical JSON becomes human-friendly
- **Structured information**: Path context preserves data relationships
- **Flexible querying**: Natural language queries work on technical data

### ⚡ **For Performance**

- **Optimal chunk sizes**: 1KB limit ensures efficient processing
- **Smart boundaries**: Semantic chunking improves relevance
- **Efficient storage**: Balanced chunk sizes optimize retrieval

## Configuration

### **Chunk Size Adjustment**

Modify the 1KB limit in `_read_json_file_chunks`:

```python
max_chunk_size = 1024  # Change to 2048 for 2KB chunks
```

### **Humanization Customization**

Customize humanization rules in `_humanize_json_object`:

```python
# Custom key formatting
human_key = key.replace('_', ' ').replace('-', ' ').title()

# Custom value processing
if isinstance(value, str) and len(value) > 100:
    human_text.append(f"{human_key} is a long text...")
```

## Testing

Use the test script to verify functionality:

```bash
python test_json_chunking.py
```

This will:
- Create sample JSON files with various structures
- Process them through the chunking pipeline
- Display humanized results
- Verify chunk size compliance
- Show before/after comparisons

The JSON chunking system transforms complex structured data into searchable, human-readable chunks while preserving semantic meaning and maintaining optimal sizes for vector storage and retrieval.