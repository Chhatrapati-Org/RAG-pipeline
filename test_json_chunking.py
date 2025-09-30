"""
Test script for JSON chunking functionality.
"""

import json
import os
from rag.merged_pipeline import MergedRAGWorker, SharedEmbeddingModel

def create_test_json_files():
    """Create test JSON files with different structures."""
    
    # Test file 1: Simple nested structure
    test_data_1 = {
        "user_profile": {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "English"
            }
        },
        "order_history": [
            {
                "order_id": "ORD-001",
                "date": "2025-01-15",
                "total": 99.99,
                "items": ["Laptop", "Mouse", "Keyboard"],
                "status": "delivered"
            },
            {
                "order_id": "ORD-002", 
                "date": "2025-02-01",
                "total": 29.99,
                "items": ["Book"],
                "status": "shipped"
            }
        ],
        "support_tickets": {
            "open": [],
            "closed": [
                {
                    "ticket_id": "TKT-123",
                    "subject": "Login issue",
                    "description": "Cannot log into my account after password reset. Tried multiple times but getting authentication error.",
                    "resolution": "Password reset link was expired. New link sent and issue resolved.",
                    "created_date": "2025-01-20",
                    "resolved_date": "2025-01-21"
                }
            ]
        }
    }
    
    # Test file 2: Large nested structure 
    test_data_2 = {
        "company": {
            "name": "TechCorp Inc.",
            "founded": 2010,
            "headquarters": "San Francisco, CA",
            "employees": 500,
            "departments": {
                "engineering": {
                    "head": "Jane Smith",
                    "team_size": 150,
                    "projects": [
                        {
                            "name": "Project Alpha",
                            "description": "A revolutionary new application that will transform how users interact with data visualization and analytics. This project involves machine learning algorithms, real-time data processing, and an intuitive user interface designed for both technical and non-technical users.",
                            "status": "in_progress",
                            "budget": 500000,
                            "timeline": "6 months"
                        },
                        {
                            "name": "Project Beta", 
                            "description": "An enterprise-grade security solution that provides comprehensive threat detection, automated response capabilities, and detailed audit trails for compliance purposes.",
                            "status": "planning",
                            "budget": 750000,
                            "timeline": "8 months"
                        }
                    ]
                },
                "marketing": {
                    "head": "Bob Johnson",
                    "team_size": 25,
                    "campaigns": [
                        "Product Launch Q1",
                        "Brand Awareness Campaign", 
                        "Customer Retention Program"
                    ]
                },
                "sales": {
                    "head": "Alice Brown",
                    "team_size": 40,
                    "quarterly_targets": {
                        "Q1": 1000000,
                        "Q2": 1200000,
                        "Q3": 1100000,
                        "Q4": 1500000
                    }
                }
            }
        }
    }
    
    # Create test directory
    test_dir = "test_json_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Write test files
    with open(os.path.join(test_dir, "user_data.json"), 'w') as f:
        json.dump(test_data_1, f, indent=2)
    
    with open(os.path.join(test_dir, "company_data.json"), 'w') as f:
        json.dump(test_data_2, f, indent=2)
    
    print(f"✅ Created test JSON files in {test_dir}/")
    return test_dir

def test_json_chunking():
    """Test the JSON chunking functionality."""
    
    print("🧪 Testing JSON Chunking Functionality")
    print("=" * 50)
    
    # Create test files
    test_dir = create_test_json_files()
    
    try:
        # Create a worker with shared model (we'll mock it for testing)
        class MockSharedModel:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]  # Mock embeddings
        
        worker = MergedRAGWorker(
            worker_id=0,
            chunk_size_kb=4,  # 4KB chunks  
            shared_model=MockSharedModel()
        )
        
        # Test each JSON file
        for filename in ["user_data.json", "company_data.json"]:
            file_path = os.path.join(test_dir, filename)
            print(f"\n📁 Processing: {filename}")
            print("-" * 30)
            
            chunk_count = 0
            for chunk_text, file_name, chunk_id in worker._read_json_file_chunks(file_path):
                chunk_count += 1
                chunk_size = len(chunk_text.encode('utf-8'))
                
                print(f"\n🧩 Chunk {chunk_id + 1} ({chunk_size} bytes):")
                print(f"   {chunk_text[:200]}..." if len(chunk_text) > 200 else f"   {chunk_text}")
                
                # Verify chunk is under 1KB as specified
                if chunk_size > 1024:
                    print(f"   ⚠️ WARNING: Chunk exceeds 1KB limit!")
                else:
                    print(f"   ✅ Size OK (under 1KB)")
            
            print(f"\n📊 Total chunks for {filename}: {chunk_count}")
        
        print(f"\n🎉 JSON chunking test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory: {test_dir}")
    except:
        pass

def test_humanization_examples():
    """Show examples of JSON humanization."""
    
    print("\n🤖 JSON Humanization Examples")
    print("=" * 40)
    
    # Example JSON structures
    examples = [
        {
            "name": "Simple Object",
            "data": {
                "user_id": 12345,
                "user_name": "john_doe",
                "is_active": True,
                "last_login": None,
                "account_balance": 150.75
            }
        },
        {
            "name": "Array Example", 
            "data": {
                "shopping_cart": [
                    {"item": "Laptop", "price": 999.99, "quantity": 1},
                    {"item": "Mouse", "price": 29.99, "quantity": 2},
                    {"item": "Keyboard", "price": 79.99, "quantity": 1}
                ]
            }
        },
        {
            "name": "Nested Structure",
            "data": {
                "company_info": {
                    "basic_details": {
                        "name": "TechCorp",
                        "founded_year": 2010,
                        "employee_count": 500
                    },
                    "contact_info": {
                        "email": "info@techcorp.com",
                        "phone": "+1-555-0123",
                        "address": {
                            "street": "123 Tech Street",
                            "city": "San Francisco",
                            "state": "CA",
                            "zip": "94105"
                        }
                    }
                }
            }
        }
    ]
    
    # Mock worker for humanization testing
    class MockWorker:
        def _humanize_json_object(self, obj, path):
            from rag.merged_pipeline import MergedRAGWorker
            real_worker = MergedRAGWorker(0, 4)
            return real_worker._humanize_json_object(obj, path)
    
    mock_worker = MockWorker()
    
    for example in examples:
        print(f"\n📋 {example['name']}:")
        print("Raw JSON:")
        print(f"   {json.dumps(example['data'], indent=2)}")
        
        print("Humanized:")
        humanized = mock_worker._humanize_json_object(example['data'], example['name'])
        print(f"   {humanized}")
        print("-" * 40)

if __name__ == "__main__":
    test_json_chunking()
    test_humanization_examples()