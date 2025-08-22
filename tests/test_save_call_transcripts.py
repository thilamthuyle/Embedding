### Case 1
# 1.1
assistant_id = "08KKUaKaiatlmSUKZKkQ"
    num_calls = 2
    save_to_dir = CALL_TRANSCRIPTS_DIR
    process_call(assistant_id, num_calls, save_to_dir)

-> expected output: 
Save 2 transcripts
9fc86493-cca7-4581-b713-9701509899d0.json
946992ef-436d-4576-9556-1719af6449f3.json

# 1.2
assistant_id = "08KKUaKaiatlmSUKZKkQ"
    num_calls = None
    save_to_dir = CALL_TRANSCRIPTS_DIR
    process_call(assistant_id, num_calls, save_to_dir)

-> Expected output:
5 transcripts saved:
a5d2d5cf-8b2d-4012-b27d-263fa3fd09fa
190ca061-21b4-4a52-8875-d8133ad0484c
3c03be07-b754-405d-8f50-5c9599183966
9fc86493-cca7-4581-b713-9701509899d0
946992ef-436d-4576-9556-1719af6449f3