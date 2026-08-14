import argparse

from .narrator import narrate
from .planner import run

SCENARIOS = {
    "clean": dict(simulate_timeout_category=None, timeout_fraction=0.0),
    "poisoned": dict(simulate_timeout_category=None, timeout_fraction=0.0),
    "failure-interaction": dict(simulate_timeout_category="electronics", timeout_fraction=0.6),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="LEC AI build assessment: injection-resistant ranking agent")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="clean")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    opts = SCENARIOS[args.scenario]
    result = run(args.scenario, **opts)

    print(f"\n=== Scenario: {args.scenario} ===\n")
    print("--- Plan events ---")
    for e in result["events"]:
        print(f"[{e.step}] {e.detail}")

    print("\n--- Top ranked items ---")
    for item in result["ranked_items"][: args.top]:
        print(
            f"#{item.product_id} {item.title[:50]!r} "
            f"(${item.price}, catalog rating {item.catalog_rating}, strategy={item.strategy_used}, "
            f"score={item.adjusted_score:.3f}) -- {'; '.join(item.trust_notes)}"
        )

    print("\n--- Narrative summary ---")
    print(narrate(result["ranked_items"], result["trust_reports"], top_n=args.top))


if __name__ == "__main__":
    main()
