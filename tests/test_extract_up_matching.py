from pathlib import Path
import json
import shutil
import sys
import collections

sys.path.insert(0, "/www/Embedding")
from src.generate_matching_inputs.extract_up_matchings import process_call_transcript
from src.generate_matching_inputs.utils import process_matching_json_file



def test_unique_ut_matching_per_call(ut_to_conv_path_dir=Path("/www/files/up_matching_dataset/inputs/ut_to_conv_path/")):
    """
    Test that each user_text is matched only once for each call transcript.
    Between 2 consecutive user_texts, if there are multiple assistant messages with different conv_path_ids, we want
    to make sure that only the first assistant message's conv_path_id is used for user_text matching.
    """
    matching_json_files = [
        file
        for file in Path(ut_to_conv_path_dir).rglob("*.json")
        if file.name != "conversation.json"
    ]

    seen_ut_matchings = set()

    for file in matching_json_files:
        assistant_id, call_id = (file.parts[-3], file.parts[-2])
        user_text_idx = json.loads(file.read_text(encoding="utf-8")).get("user_text_idx")
        assert (assistant_id, call_id, user_text_idx) not in seen_ut_matchings, (
            f"User text index {user_text_idx} for assistant {assistant_id} and call {call_id} has appeared more than once."
        )
        seen_ut_matchings.add((assistant_id, call_id, user_text_idx))


def test_process_call_transcript():
    Input = collections.namedtuple("Input", ["save_to_dir", "language", "assistant_id", "call_id"])
    Output = collections.namedtuple("num_matchings", "candidates")
    call_transcripts_dir = "/www/files/call_transcripts"
    save_matching_to_dir = Path("/www/files/test/")
    test_cases = []
    expected_matchings = []

    # TEST CASE 1:
    # conv_path_id with §NO_NEED§ user_proppt candidates + consecutive ASSISTANT messages with different conv_path_ids (to check correctness of user_text_idx)
    language = "es"
    assistant_id = "nRvPgRrKr2MuO7bjYCRY"
    call_id = "255edbe5-cd84-4969-89a0-269a31437e75"
    test_cases.append(Input(save_matching_to_dir, language, assistant_id, call_id))
    expected_matchings.append(
        Output(
            1,
            [
                {
                    "language": "es",
                    "assistant_id": "nRvPgRrKr2MuO7bjYCRY",
                    "call_id": "255edbe5-cd84-4969-89a0-269a31437e75",
                    "user_text": "Hola, buenos días, pues verás, tengo un problema.",
                    "user_text_idx": 3,
                }
            ],
        )
    )

    # TEST CASE 2:
    #
    language = "fr"
    assistant_id = "7pSWfUtTqkAzZ6BvbQXB"
    call_id = "b206caaf-4278-4550-bc16-210290a8b10d"
    test_cases.append(Input(save_matching_to_dir, language, assistant_id, call_id))
    expected_matchings.append(
        Output(
            2,
            [
                {
                    "language": "fr",
                    "assistant_id": "7pSWfUtTqkAzZ6BvbQXB",
                    "call_id": "b206caaf-4278-4550-bc16-210290a8b10d",
                    "user_text": "Axima",
                    "user_text_idx": 8,
                }
            ],
        )
    )

    # TEST CASE 3: no matching found

    for test_case in test_cases:
        print("Processing test case:", test_case)
        language = test_case.language
        assistant_id = test_case.assistant_id
        call_id = test_case.call_id
        output_dir = Path(f"{save_matching_to_dir}/{language}/{assistant_id}/{call_id}")
        call_transcript_path = Path(f"{call_transcripts_dir}/{assistant_id}/{call_id}.json")

        process_call_transcript(call_transcript_path, output_dir, language, assistant_id, call_id)

        assert (
            len(list(output_dir.glob("*.json"))) == 2
        )  # expecting 2 JSON files: one is conversation.json and one for the matching
        matching = json.loads(Path(f"{output_dir}/0.json").read_text(encoding="utf-8"))

        expected_matching = {
            "language": "es",
            "assistant_id": "nRvPgRrKr2MuO7bjYCRY",
            "call_id": "255edbe5-cd84-4969-89a0-269a31437e75",
            "user_text": "Hola, buenos días, pues verás, tengo un problema.",
            "user_text_idx": 3,
        }
        expected_conv_path_id = "73077eff3efe60a2e3cb899b74f33b94_3404de73309c5df091f22ed6ac9fbe58_b11e96af3ced0c747c8974b45929d3fb_bed9928d9263f86767cbb3d8ab1f2e06"
        expected_source_node_id = expected_conv_path_id.split("_")[0]

        # verify that all candidate conv_paths have expected_source_node_id as source_node_id
        for key in expected_matching.keys():
            assert (
                matching[key] == expected_matching[key]
            ), f"Mismatch in {key}: expected {expected_matching[key]}, got {matching[key]}"
        for conv_path_id in matching["candidates"]["conv_path_id"]:
            source_node_id = conv_path_id.split("_")[0]
            assert (
                source_node_id == expected_source_node_id
            ), f"Mismatch in conv_path_id: expected {expected_source_node_id}, got {source_node_id}"

        # delete the output_dir after test
        shutil.rmtree("/www/files/test")


def test_save_up_to_examples_matching():
    ### 1
    inputs_dir = Path(f"/www/files/matching_dataset/inputs")
    candidates = {
        "up": [
            "[Speaker asks if payments have been made]",
            "[Speaker asks what is the next payment date]",
            "[Speaker says they don't have any more doubts]",
            "[Speaker asks I would like to know whether the allowances have already been paid or not]",
        ],
        "aa": [
            "\u00a7LLM\u00a7",
            "\u00a7LLM\u00a7",
            "I hope I was able to help. Thank you for calling Kidslife.",
            "\u00a7LLM\u00a7",
        ],
        "aq": [
            "Is there something else you're unsure about?",
            "Is there something else you're unsure about?",
            "Have a good rest of your day bye.",
            "Is there something else you're unsure about?",
        ],
        "conv_path_id": [
            "7e201a0910d1b3bdf6557a7cc1b4e583_e1389bc01ff8db553e8d4cb1d94bedfa_7fc5cedf72e688a381e5e727ffbf17c2_cecb03c867f500c3205ca3b2efbbb067",
            "7e201a0910d1b3bdf6557a7cc1b4e583_d604afe68e05ee33ed64ea676ae7b673_7fc5cedf72e688a381e5e727ffbf17c2_cecb03c867f500c3205ca3b2efbbb067",
            "7e201a0910d1b3bdf6557a7cc1b4e583_29622d615a1ccaea49c135f2fa3d0a91_a1e254119d3b7e3ea0ea071a3c2434da_3e8c3859a2ef4e2405dd10ffceba4c14",
            "7e201a0910d1b3bdf6557a7cc1b4e583_c7a20faeac33f0b9ea2878aef6e286d7_7fc5cedf72e688a381e5e727ffbf17c2_cecb03c867f500c3205ca3b2efbbb067",
        ],
    }
    save_up_to_examples_matching(Path(f"{inputs_dir}/up_to_examples"), candidates)

    ### 2
    up_to_examples_dir = "/www/files/matching_dataset/inputs/up_to_examples"
    candidates = {
        "up": [
            "[Speaker says yes without mentioning the countries]",
            "[Speaker says yes without mentioning the countries]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says no]",
            "[Speaker says yes]",
            "[Speaker says yes]",
            "[Speaker says yes]",
            "[Speaker dit non notre agence n'est pas pas active \u00e0 l'international]",
            "[Speaker dit non notre agence n'est pas pas active \u00e0 l'international]",
            "[Speaker dit non notre agence n'est pas pas active \u00e0 l'international]",
            "[Speaker dit non notre agence n'est pas pas active \u00e0 l'international]",
            "[Speaker dit non notre agence n'est pas pas active \u00e0 l'international]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "[Speaker says they are not active in several countries]",
            "\u00a7NO\u00a7",
            "\u00a7NO\u00a7",
            "\u00a7NO\u00a7",
            "[Speaker says they are active in several countries, and mentions the countries]",
            "[Speaker says they are active in several countries, and mentions the countries]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "[Speaker says yes with details]",
            "\u00a7YES\u00a7",
            "\u00a7YES\u00a7",
            "\u00a7YES\u00a7",
            "\u00a7YES\u00a7",
            "\u00a7YES\u00a7",
            "[Speaker says that they are not active in several countries]",
            "[Speaker says that they are not active in several countries]",
            "[Speaker says that they are not active in several countries]",
            "[Speaker says that they are not active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says they are active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker says their agency is not active in several countries]",
            "[Speaker say nope]",
            "[Speaker say nope]",
            "[Speaker say nope]",
            "[Speaker say nope]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker says that their agency is active in several countries]",
            "[Speaker dit oui avec d\u00e9tails]",
            "[Speaker dit oui avec d\u00e9tails]",
            "[Speaker dit oui avec d\u00e9tails]",
            "[Speaker dit oui avec d\u00e9tails]",
            "[speaker says nope]",
            "[Speaker saussss]",
            "[Speaker saussss]",
            "[Speaker saussss]",
            "[Speaker saussss]",
            "[Speaker says yes, we've already invested in other platforms].",
        ],
        "aa": [
            "And...",
            "And...",
            "Unqualified.",
            "Unqualified.",
            "Unqualified.",
            "Unqualified.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Merci pour votre temps. Je vais vous envoyer un email avec plus d\u2019informations sur SortList et nos webinaires pour vous aider \u00e0 en savoir plus.",
            "Thank you for your time. I'll send you an email with more information about SortList and our webinars to help you learn more.",
            "Amazing! We'll send you an email with information about our regular webinars \u2014 They\u2019ll help you better understand how the platform works and guide you in creating a well-structured profile...",
            "Alright. And...",
            "Alright. And...",
            "Alright. And...",
            "Merci pour vos r\u00e9ponses. Je vais maintenant organiser un rendez-vous avec l\u2019un de nos Business Consultants pour vous accompagner au mieux.",
            "Merci pour vos r\u00e9ponses. Je vais maintenant organiser un rendez-vous avec l\u2019un de nos Business Consultants pour vous accompagner au mieux.",
            "Merci pour vos r\u00e9ponses. Je vais maintenant organiser un rendez-vous avec l\u2019un de nos Business Consultants pour vous accompagner au mieux.",
            "Merci pour vos r\u00e9ponses. Je vais maintenant organiser un rendez-vous avec l\u2019un de nos Business Consultants pour vous accompagner au mieux.",
            "Merci pour vos r\u00e9ponses. Je vais maintenant organiser un rendez-vous avec l\u2019un de nos Business Consultants pour vous accompagner au mieux.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "Merci pour votre temps. Je vais vous envoyer un email avec plus d\u2019informations sur SortList et nos webinaires pour vous aider \u00e0 en savoir plus.",
            "Thank you for your time. I'll send you an email with more information about SortList and our webinars to help you learn more.",
            "Amazing! We'll send you an email with information about our regular webinars \u2014 They\u2019ll help you better understand how the platform works and guide you in creating a well-structured profile...",
            "Alright. And...",
            "Alright. And...",
            "Allright.... And...",
            "Allright.... And...",
            "Allright.... And...",
            "Alright. And...",
            "Alright. And...",
            "Alright. And...",
            "D'accord... Et,,,.",
            "D'accord... Et,,,.",
            "D'accord... Et,,,.",
            "Alright...",
            "Alright...",
            "Alright. And...",
            "Alright. And...",
            "Alright. And...",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Alright. And...",
            "Alright. And...",
            "Alright. And...",
            "Merci pour votre temps. Je vais vous envoyer un email avec plus d\u2019informations sur SortList et nos webinaires pour vous aider \u00e0 en savoir plus.",
            "Thank you for your time. I'll send you an email with more information about SortList and our webinars to help you learn more.",
            "Amazing! We'll send you an email with information about our regular webinars \u2014 They\u2019ll help you better understand how the platform works and guide you in creating a well-structured profile...",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "Unqualified.",
            "Unqualified.",
            "Unqualified.",
            "Unqualified.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers. I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "Thank you for your answers.",
            "D'accord... Et,,,.",
            "D'accord... Et,,,.",
            "D'accord... Et,,,.",
            "D'accord... Et,,,.",
            "???",
            "But how.",
            "But how.",
            "But how.",
            "But how.",
            "Alright...",
        ],
        "aq": [
            "In which countries do you operate?",
            "Is your company active in several countries? If so, which ones?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "Could you please confirm your email address for me?",
            "Have a great day.",
            "Could you please confirm your email address for me?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "Quelle est votre disponibilit\u00e9 ?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.Sounds good?",
            "Could you please confirm your email address for me?",
            "Have a great day.",
            "Could you please confirm your email address for me?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "In which countries do you operate in?",
            "Is your company active in several countries? If so, which ones?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Could you please confirm your email address for me?",
            "Have a great day.",
            "Could you please confirm your email address for me?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.Sounds good?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support.Sounds good?",
            "Avez-vous des locaux physiques dans d\u2019autres pays ?",
            "Do you have premises in other countries?",
            "Do you have offices in those countries?",
            "Do you have offices in other countries?",
            "Have a great day.",
            "Sounds good?",
            "I will now arrange a meeting with one of our Business Consultants to provide you with the best possible support, sounds good?",
            "Would you like me to help you schedule an appointment?",
            "What's your availability?",
            "In which countries do you operate in?",
        ],
        "conv_path_id": [
            "7677124face85d1f94132e0fb4310025_0b803dfb132ed3488ff2bb179529f24c_7e556b9651643289d7385ee6593f23a1_5b76adb2a1ce142c6f414e3474ad46dd",
            "7677124face85d1f94132e0fb4310025_0b803dfb132ed3488ff2bb179529f24c_7e556b9651643289d7385ee6593f23a1_7677124face85d1f94132e0fb4310025",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_1274291a280f030c8c3edc12c7aa604a_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_1274291a280f030c8c3edc12c7aa604a_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_1274291a280f030c8c3edc12c7aa604a_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_1274291a280f030c8c3edc12c7aa604a_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_2362c5fd18964355b7c8dd4ef2a7d82e_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_2362c5fd18964355b7c8dd4ef2a7d82e_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_2362c5fd18964355b7c8dd4ef2a7d82e_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_2362c5fd18964355b7c8dd4ef2a7d82e_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_e0ce97f881d585295e24911f3e8be76c_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_fc08d1feb57d31af532e2c4c9b5da002_56f1d431d610337ccf6542573647fde5",
            "7677124face85d1f94132e0fb4310025_0cb8be5f76b40cc4d2f53220805cc842_fc7e47fbd57c37095198bf913eb1d525_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_19a7e008df7b2a99eb6887d4367a1479_9610eee12264774df542a0132c39ce9d_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_19a7e008df7b2a99eb6887d4367a1479_9610eee12264774df542a0132c39ce9d_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_19a7e008df7b2a99eb6887d4367a1479_9610eee12264774df542a0132c39ce9d_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_27034f00cf239f410c7f25e8ab177dd1_98ed73c304820accdb2da8bbd2f10906_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_27034f00cf239f410c7f25e8ab177dd1_98ed73c304820accdb2da8bbd2f10906_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_27034f00cf239f410c7f25e8ab177dd1_98ed73c304820accdb2da8bbd2f10906_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_27034f00cf239f410c7f25e8ab177dd1_98ed73c304820accdb2da8bbd2f10906_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_27034f00cf239f410c7f25e8ab177dd1_98ed73c304820accdb2da8bbd2f10906_c35d9e9f6cc3325f8657bc309dec220d",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_2362c5fd18964355b7c8dd4ef2a7d82e_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_2362c5fd18964355b7c8dd4ef2a7d82e_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_2362c5fd18964355b7c8dd4ef2a7d82e_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_2362c5fd18964355b7c8dd4ef2a7d82e_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_c4d94291ac0bf9fc8bb45eb026feedf1_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_c4d94291ac0bf9fc8bb45eb026feedf1_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_2ce9c462c8f42cba88692213f369b8fd_c4d94291ac0bf9fc8bb45eb026feedf1_bc2ced4fee4523bd95fee09d584a7357",
            "7677124face85d1f94132e0fb4310025_4adfa89c8745790a0268df2f1f57f999_e0ce97f881d585295e24911f3e8be76c_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_4adfa89c8745790a0268df2f1f57f999_fc08d1feb57d31af532e2c4c9b5da002_56f1d431d610337ccf6542573647fde5",
            "7677124face85d1f94132e0fb4310025_4adfa89c8745790a0268df2f1f57f999_fc7e47fbd57c37095198bf913eb1d525_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_5fdc57e5993c6e9151f9f3892c11d0c8_9610eee12264774df542a0132c39ce9d_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_5fdc57e5993c6e9151f9f3892c11d0c8_9610eee12264774df542a0132c39ce9d_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_62fb3f3b95309b505795f4aa97ccd418_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_62fb3f3b95309b505795f4aa97ccd418_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_62fb3f3b95309b505795f4aa97ccd418_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_9610eee12264774df542a0132c39ce9d_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_9610eee12264774df542a0132c39ce9d_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_9610eee12264774df542a0132c39ce9d_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_ebc34c977aebd4d36a51e7fe8bf77aa2_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_ebc34c977aebd4d36a51e7fe8bf77aa2_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_67866ba89a946954e35aede435a2fef0_ebc34c977aebd4d36a51e7fe8bf77aa2_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_6a5029071f8a570a59f9361a8da6dee7_2ef68f07babcef44e7b3f7ef2e94b824_603da86fe72b84540ce35d1554c684bf",
            "7677124face85d1f94132e0fb4310025_6a5029071f8a570a59f9361a8da6dee7_2ef68f07babcef44e7b3f7ef2e94b824_7677124face85d1f94132e0fb4310025",
            "7677124face85d1f94132e0fb4310025_6a5029071f8a570a59f9361a8da6dee7_9610eee12264774df542a0132c39ce9d_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_6a5029071f8a570a59f9361a8da6dee7_9610eee12264774df542a0132c39ce9d_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_6a5029071f8a570a59f9361a8da6dee7_9610eee12264774df542a0132c39ce9d_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_8e5bb4e089c75d1f8d39d2a9bfdf6b79_2362c5fd18964355b7c8dd4ef2a7d82e_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_8e5bb4e089c75d1f8d39d2a9bfdf6b79_2362c5fd18964355b7c8dd4ef2a7d82e_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_8e5bb4e089c75d1f8d39d2a9bfdf6b79_2362c5fd18964355b7c8dd4ef2a7d82e_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_8e5bb4e089c75d1f8d39d2a9bfdf6b79_2362c5fd18964355b7c8dd4ef2a7d82e_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_9610eee12264774df542a0132c39ce9d_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_9610eee12264774df542a0132c39ce9d_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_9610eee12264774df542a0132c39ce9d_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_e0ce97f881d585295e24911f3e8be76c_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_fc08d1feb57d31af532e2c4c9b5da002_56f1d431d610337ccf6542573647fde5",
            "7677124face85d1f94132e0fb4310025_9fef6109b757d33a0ed44c94e512fd13_fc7e47fbd57c37095198bf913eb1d525_7e3518c206673d1ed2ccd6ae9840f8cc",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_2362c5fd18964355b7c8dd4ef2a7d82e_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_2362c5fd18964355b7c8dd4ef2a7d82e_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_2362c5fd18964355b7c8dd4ef2a7d82e_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_2362c5fd18964355b7c8dd4ef2a7d82e_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_c4d94291ac0bf9fc8bb45eb026feedf1_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_c4d94291ac0bf9fc8bb45eb026feedf1_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_a02705a6dda088a9c3b62637594bddd5_c4d94291ac0bf9fc8bb45eb026feedf1_bc2ced4fee4523bd95fee09d584a7357",
            "7677124face85d1f94132e0fb4310025_a9178025f9ba84878a45760c0a3a0f47_1274291a280f030c8c3edc12c7aa604a_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_a9178025f9ba84878a45760c0a3a0f47_1274291a280f030c8c3edc12c7aa604a_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_a9178025f9ba84878a45760c0a3a0f47_1274291a280f030c8c3edc12c7aa604a_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_a9178025f9ba84878a45760c0a3a0f47_1274291a280f030c8c3edc12c7aa604a_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_2362c5fd18964355b7c8dd4ef2a7d82e_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_2362c5fd18964355b7c8dd4ef2a7d82e_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_2362c5fd18964355b7c8dd4ef2a7d82e_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_2362c5fd18964355b7c8dd4ef2a7d82e_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_c4d94291ac0bf9fc8bb45eb026feedf1_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_c4d94291ac0bf9fc8bb45eb026feedf1_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_c1921a22b5a3ccde8ad7566950b88e32_c4d94291ac0bf9fc8bb45eb026feedf1_bc2ced4fee4523bd95fee09d584a7357",
            "7677124face85d1f94132e0fb4310025_c2878d75cbf016869c5a8acb5b05b246_ebc34c977aebd4d36a51e7fe8bf77aa2_7c5b7701c0028ed1ffbcc8def23d5a80",
            "7677124face85d1f94132e0fb4310025_c2878d75cbf016869c5a8acb5b05b246_ebc34c977aebd4d36a51e7fe8bf77aa2_97f6ffe2de46b112de00bc0ab26d5fb1",
            "7677124face85d1f94132e0fb4310025_c2878d75cbf016869c5a8acb5b05b246_ebc34c977aebd4d36a51e7fe8bf77aa2_b74016a4864cba73e96a830345237ed7",
            "7677124face85d1f94132e0fb4310025_c2878d75cbf016869c5a8acb5b05b246_ebc34c977aebd4d36a51e7fe8bf77aa2_de7881b2ed0f69229ba42b0522ea9046",
            "7677124face85d1f94132e0fb4310025_c7a4b3ba620aea638a9493e51a9ef4b7_4609cb7c7bc655a904d96703eeadc2ae_56f1d431d610337ccf6542573647fde5",
            "7677124face85d1f94132e0fb4310025_d1d1550a1ed131af0cf697289e6cf37c_0f609e67385948fc663b5588aced7674_350d291d6d874c03aea8e3d78639da2a",
            "7677124face85d1f94132e0fb4310025_d1d1550a1ed131af0cf697289e6cf37c_0f609e67385948fc663b5588aced7674_45af50bfcfd98e2c666b9b3aa01d4025",
            "7677124face85d1f94132e0fb4310025_d1d1550a1ed131af0cf697289e6cf37c_0f609e67385948fc663b5588aced7674_4cecbfa85ebc5c0361513f7dc66aff42",
            "7677124face85d1f94132e0fb4310025_d1d1550a1ed131af0cf697289e6cf37c_0f609e67385948fc663b5588aced7674_ad33c04f38d2d7bff0a9fbc1b8cb495f",
            "7677124face85d1f94132e0fb4310025_e6d5e1f2a1a9134189fc6b9a3dbcb859_2ef68f07babcef44e7b3f7ef2e94b824_603da86fe72b84540ce35d1554c684bf",
        ],
    }
    process_matching_json_file(up_to_examples_dir, candidates)


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
