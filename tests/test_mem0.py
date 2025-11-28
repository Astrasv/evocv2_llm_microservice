import time
import os
from mem0 import MemoryClient
from app.config import settings


API_KEY = settings.mem0_api_key
USER_ID = "test_user_lifecycle_v2" 

# Initialize Client
try:
    print(f"🔌 Connecting to Mem0 with User ID: {USER_ID}...")
    client = MemoryClient(api_key=API_KEY)
except Exception as e:
    print(f"  CRITICAL: Failed to initialize client. {e}")
    exit(1)

def print_separator(step_name):
    print(f"\n{'='*10} {step_name} {'='*10}")

print_separator("STEP 1: ADD MEMORY")
messages = [
    {"role": "user", "content": "I am testing the system. My favorite color is #FF5733 (Persimmon Orange)."}
]

try:
    add_response = client.add(messages, user_id=USER_ID)
    print("    Add Request Sent.")
    print(f"   Response keys: {add_response.keys()}")
    
    print(" Waiting 3 seconds for indexing...")
    time.sleep(3)
except Exception as e:
    print(f"  STEP 1 FAILED: {e}")
    exit(1)


print_separator("STEP 2: GET ALL")
target_memory_id = None

try:
    # CRITICAL: V2 API requires 'filters' dict, not direct user_id arg
    response = client.get_all(filters={"user_id": USER_ID})
    memories = response.get("results", [])

    if not memories:
        print("   WARNING: No memories found. Indexing might be slow or extraction failed.")
        exit(1)
    
    print(f"    Found {len(memories)} memories.")
    for mem in memories:
        print(f"   - [ID: {mem['id']}] {mem['memory']}")
        if "Orange" in mem['memory']:
            target_memory_id = mem['id']

    if not target_memory_id:
        # Fallback: just pick the first one to test update/delete
        target_memory_id = memories[0]['id']
        print(f"   (Selected first memory ID for testing: {target_memory_id})")

except Exception as e:
    print(f"  STEP 2 FAILED: {e}")
    exit(1)


print_separator("STEP 3: SEARCH")
query = "What is my favorite color?"

try:
    # CRITICAL: Search also requires 'filters' in V2
    search_res = client.search(query, filters={"user_id": USER_ID})
    results = search_res.get("results", [])
    
    if results:
        print(f"    Search Successful for query '{query}':")
        for res in results:
            print(f"   - {res['memory']} (Score: {res.get('score', 'N/A')})")
    else:
        print("  Search returned no results.")
except Exception as e:
    print(f"STEP 3 FAILED: {e}")

print_separator("STEP 4: UPDATE")
new_text = "I am testing the system. My favorite color is actually Blue."

try:
    # Update requires the specific MEMORY_ID, not a filter
    client.update(memory_id=target_memory_id, data=new_text)
    print(f"    Update Request Sent for ID: {target_memory_id}")
    
    print(" Waiting 2 seconds for update propagation...")
    time.sleep(2)
    
    # Verify update
    check_response = client.get(target_memory_id)
    print(f"   Current Value: {check_response.get('memory')}")
except Exception as e:
    print(f"  STEP 4 FAILED: {e}")


print_separator("STEP 5: DELETE")

try:
    # Delete requires the specific MEMORY_ID
    client.delete(memory_id=target_memory_id)
    print(f"    Delete Request Sent for ID: {target_memory_id}")
    
    print("Verifying deletion...")
    time.sleep(2)
    
    # Verify it's gone
    final_check = client.get(target_memory_id)
    # Note: Depending on API version, get() might raise 404 or return None/Empty
    if not final_check: 
        print("    Confirmed: Memory is gone.")
    else:
        print(f"   Warning: Memory might still exist (API returned: {final_check})")
        
except Exception as e:
    # If API raises an error for missing ID, that is actually a success for deletion
    if "not found" in str(e).lower() or "404" in str(e):
        print("    Confirmed: Memory not found (Deletion successful).")
    else:
        print(f"  STEP 5 FAILED: {e}")

print_separator("TEST COMPLETE")