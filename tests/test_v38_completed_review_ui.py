"""Real queue acceptance and readback, replacing the legacy helper-injected gate."""
def test_completed_review_lifecycle_with_real_frontend(review_queue_results):
    for case in (
        "explicit restart consumed and Chunk 1 Continue restored",
        "850 ms terminal readback releases setup after it resolves",
        "same completed revision reuse exits setup without new Sampling",
        "three review queues advance without any executed events",
        "settings edited during execution remain out of date after readback",
        "same setup request is not mutated by legacy preparation then API preparation",
    ):
        assert review_queue_results[case]["pass"]
