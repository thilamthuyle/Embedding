from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, "/www/Embedding")
from src.generate_matching_inputs.extract_up_matchings import process_call_transcript


def test_process_call_transcript():
    # case 1
    call_transcript_path = Path('/www/files/call_transcripts/nRvPgRrKr2MuO7bjYCRY/255edbe5-cd84-4969-89a0-269a31437e75.json')
    output_dir = Path('/www/files/test/nRvPgRrKr2MuO7bjYCRY/255edbe5-cd84-4969-89a0-269a31437e75')
    language = 'en'
    assistant_id = 'nRvPgRrKr2MuO7bjYCRY'
    call_id = '255edbe5-cd84-4969-89a0-269a31437e75'
    
    process_call_transcript(call_transcript_path, output_dir, language, assistant_id, call_id)
    assert len(list(output_dir.glob("*.json"))) == 2 # expecting 2 JSON files: one is conversation.json and one for the matching
    matching = json.loads(Path(f"{output_dir}/0.json").read_text(encoding='utf-8'))
    
    expected_matching = {
        "language": "en",
        "assistant_id": "nRvPgRrKr2MuO7bjYCRY",
        "call_id": "255edbe5-cd84-4969-89a0-269a31437e75",
        "user_text": "Hola, buenos días, pues verás, tengo un problema.",
        "user_text_idx": 3,
    }
    expected_conv_path_id = '73077eff3efe60a2e3cb899b74f33b94_3404de73309c5df091f22ed6ac9fbe58_b11e96af3ced0c747c8974b45929d3fb_bed9928d9263f86767cbb3d8ab1f2e06'
    expected_source_node_id = expected_conv_path_id.split("_")[0]
    
    # verify that all candidate conv_paths have expected_source_node_id as source_node_id
    for key in expected_matching.keys():
        assert matching[key] == expected_matching[key], f"Mismatch in {key}: expected {expected_matching[key]}, got {matching[key]}"
    for conv_path_id in matching["candidates"]["conv_path_id"]:
        source_node_id = conv_path_id.split("_")[0]
        assert source_node_id == expected_source_node_id, f"Mismatch in conv_path_id: expected {expected_source_node_id}, got {source_node_id}"

    # delete the output_dir after test
    shutil.rmtree("/www/files/test")



# ####
# # call_transcript with repetitive conv_path_id (with conversation go back into a previously visited node)
# assistant_id = 'nRvPgRrKr2MuO7bjYCRY'
# call_id = '255edbe5-cd84-4969-89a0-269a31437e75'

# conv_path_id_idx = [4, 5, 7, 8, 10, 11, 12]
# conv_path_ids = [
#  '73077eff3efe60a2e3cb899b74f33b94_3404de73309c5df091f22ed6ac9fbe58_b11e96af3ced0c747c8974b45929d3fb_bed9928d9263f86767cbb3d8ab1f2e06',
#  '73077eff3efe60a2e3cb899b74f33b94_3404de73309c5df091f22ed6ac9fbe58_b11e96af3ced0c747c8974b45929d3fb_bed9928d9263f86767cbb3d8ab1f2e06',
#  'bed9928d9263f86767cbb3d8ab1f2e06_e416d7315594a080bbd5397a163dc8ed_7d1fdf38aea980d8fdd72c3506d5cdc9_0f78760474867898531dda8b38b11faf',
#  'bed9928d9263f86767cbb3d8ab1f2e06_e416d7315594a080bbd5397a163dc8ed_7d1fdf38aea980d8fdd72c3506d5cdc9_0f78760474867898531dda8b38b11faf',
#  '0f78760474867898531dda8b38b11faf_d74f719e2eb67295dbda5858d6866aa4_300381de2cbb37f866ddc4f888e4861f_08b92e58-31a8-45ea-9e49-55a85ec175d4',
#  '08b92e58-31a8-45ea-9e49-55a85ec175d4_a80948c904be4f37ebc4bb42c58e624d_33014647bdedb1a7d2d5c4812fe6d1c2_b0bd478e6cd83fe97b08353d1de39a37',
#  '0f78760474867898531dda8b38b11faf_d74f719e2eb67295dbda5858d6866aa4_300381de2cbb37f866ddc4f888e4861f_08b92e58-31a8-45ea-9e49-55a85ec175d4'
# ]

# user_text_ids = [3, 6, 9]
# user_texts = [
#     'Hola, buenos días, pues verás, tengo un problema.',
#     'Soy un particular.',
#     'Sí, puedes utilizarlo.',
# ]

# Observations:
# - 2 consecutive conv_path_ids with different source_nodes. Check that!!!!

    



# # Test extract_matching_candidates_from_source_node from utils.py
# from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
# from utils import ConvPath

# namedtupple = source_node_id, one_conv_path_with_source_node
# ("11Cfhpb2sRSmDDhOcMZwj", "11Cfhpb2sRSmDDhOcMZwj_P6DJfYiWyUe733FzoXj9_y5nqeMleaXkEoZs") # one among possible conv_path_ids has target_node_id = None
# ("6f9f88344d8d9ed41aa94da2644e121c", "6f9f88344d8d9ed41aa94da2644e121c_a9226fc4c9693896251ff045797b9d27_76fb41c902d6f8cb30f7a6bbcb7fc20c_bc05ab88b87e67c06cadf44895c07c1f") 


# source_node_id = "11Cfhpb2sRSmDDhOcMZwj"
# conv_paths_from_source_nodes_dict = get_conv_paths_from_source_nodes_dict(
#     [ConversationalPaths.get("11Cfhpb2sRSmDDhOcMZwj_P6DJfYiWyUe733FzoXj9_y5nqeMleaXkEoZs")]
# )
# extract_matching_candidates_from_source_node(source_node_id, conv_paths_from_source_nodes_dict)




# # Test save_matching_to_json from utils.py
# save_matching_to_json(output_dir=Path(
#             "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
#         ),
#         language="en",
#         assistant_id="d37mNMstUaZwSPqtXUIJ",
#         call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
#         matching_id=0, 
#         user_text="What is the weather like today?",
#         user_text_idx=0,  # Assuming the first message is the user text
#         candidates=candidates
#     )


# ### Test process_call_transcript in extract_matching_from_call_transcripts.py
# path = Path(
#         "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
#     )
# all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
# messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
#     all_messages
# )
# depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
# conv_paths_from_source_nodes = get_conv_paths_from_source_nodes_dict(
#     list(depth2_conv_paths_by_ids_dict.values())
# )
# process_call_transcript(
#     call_transcript_path,
#     output_dir=Path(
#         "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
#     ),
#     language="en",
#     assistant_id="d37mNMstUaZwSPqtXUIJ",
#     call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
# )
# print(conv_paths_from_source_nodes)


####
 # ut_query = "Vale, pero hacerlo rápido."
    # user_prompt_id = "9874e740d0055bc52a241fc42c1a0602"
    # print(check_normalized_text_matching(ut_query, user_prompt_id))

    # call_transcript_path = Path("/www/files/call_transcripts/bzzd3IuANPMYYsitiBlo/12c8a927-94be-48fe-9fe0-c626f38584a5.json")
    # output_dir = Path("/www/files/matching_dataset/inputs/ut_to_conv_path/test/bzzd3IuANPMYYsitiBlo/12c8a927-94be-48fe-9fe0-c626f38584a5")
    # process_call_transcript(call_transcript_path, output_dir, "test", "bzzd3IuANPMYYsitiBlo", "12c8a927-94be-48fe-9fe0-c626f38584a5")

    # call_transcript_path = Path(
    #     "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
    # )
    # output_dir = Path(
    #     "/www/files/matching_dataset/inputs/ut_to_conv_path/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    # )
    # language = "en"
    # assistant_id = "d37mNMstUaZwSPqtXUIJ"
    # call_id = "ef7501dd-e530-4e70-82bd-6795b17cedc6"
    # process_call_transcript(call_transcript_path, output_dir, language, assistant_id, call_id)

    ######
    # from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
    # from utils import ConvPath

    # source_node_id = "6f9f88344d8d9ed41aa94da2644e121c"
    # conv_paths_from_source_nodes_dict = get_conv_paths_from_source_nodes_dict(
    #     [ConversationalPaths.get("6f9f88344d8d9ed41aa94da2644e121c_a9226fc4c9693896251ff045797b9d27_76fb41c902d6f8cb30f7a6bbcb7fc20c_bc05ab88b87e67c06cadf44895c07c1f")]
    # )
    # candidates = extract_matching_candidates_from_source_node(source_node_id, conv_paths_from_source_nodes_dict)
    # save_matching_to_json(output_dir=Path(
    #         "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    #     ),
    #     language="en",
    #     assistant_id="d37mNMstUaZwSPqtXUIJ",
    #     call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
    #     matching_id=0,
    #     user_text="What is the weather like today?",
    #     user_text_idx=0,  # Assuming the first message is the user text
    #     candidates=candidates
    # )

    ######
    # call_transcript_path = Path(
    #     "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
    # )
    # all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    # messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
    #     all_messages
    # )
    # depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
    # conv_paths_from_source_nodes = get_conv_paths_from_source_nodes_dict(
    #     list(depth2_conv_paths_by_ids_dict.values())
    # )
    # process_call_transcript(
    #     call_transcript_path,
    #     output_dir=Path(
    #         "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    #     ),
    #     language="en",
    #     assistant_id="d37mNMstUaZwSPqtXUIJ",
    #     call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
    # )
    # print(conv_paths_from_source_nodes)



