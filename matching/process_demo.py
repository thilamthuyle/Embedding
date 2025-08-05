import concurrent.futures
import logging

a = 1


def process_demo(i):
    return a + i


def extract_ut_to_conv_path_matching():
    """
    Extract user text to conversational path matchings from call transcripts.
    Saves the results to a specified directory using multiprocessing.
    """
    tasks = []
    for i in range:
        tasks.append((i,))

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_demo, *args) for args in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error in process_json_file: {exc}")



if __name__ == "__main__":
    extract_ut_to_conv_path_matching()
