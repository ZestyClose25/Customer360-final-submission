from pathlib import Path
from typing import Callable
from datetime import datetime
import json
import time
import requests


SERVER_URL = "http://127.0.0.1:8000"

PATH = Path(__file__).resolve().parents[0] / "data"


def stream_history(scenario: str):
    history_seed_file = (
        PATH
        / f"{scenario}"
        / "history_seed.jsonl"
    )

    entities = PATH / f"{scenario}/entities.json"
    with open(entities, "r", encoding="utf-8") as file:
        CUSTOMER = json.load(file)

    if not history_seed_file.is_file():
        raise FileNotFoundError(f"history_seed.jsonl not found for {scenario}: {history_seed_file}")

    with open(history_seed_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            event = json.loads(line)

            response = requests.post(
                f"{SERVER_URL}/events",
                json={
                    "event": event,
                    "customer" : CUSTOMER,
                    "is_live": False
                }
            )

            response.raise_for_status()

            # print(f"[HISTORY] {line_number} sent: f{event['event_id']}")

    return True

class LiveEventStreamPlay:
    def __init__(
            self,
            scenario: str,
            publish_event: Callable[[dict], None]
        ):
        self.stream_file = PATH/ f"{scenario}" / "live_stream.jsonl"
        self.replay_config_file = PATH / f"{scenario}" / "replay_config.json"
        self.publish_event = publish_event

        self.config = self._load_config()

        self.speed = self.config["replay_speed_seconds_per_simulated_day"]
        with open(PATH/f"{scenario}"/"entities.json", "r", encoding="utf-8") as file:
            self.CUSTOMER = json.load(file)

    def _load_config(self) -> dict:
        with open(self.replay_config_file, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _parse_time(timestamp: str) ->datetime:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def _calculate_delay(
            self,
            previous_time: datetime,
            current_time: datetime,
    ) -> float:
        simulated_seconds = (
            current_time - previous_time
        ).total_seconds()

        simulated_days = simulated_seconds / (24*60*60)

        real_seconds = simulated_days * self.speed
        return max(real_seconds, 0)

    def replay(self):
        previous_event_time = None

        with open(self.stream_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                event = json.loads(line)

                current_event_time = self._parse_time(event["event_time"])

                if previous_event_time is not None:
                    delay = self._calculate_delay(previous_event_time, current_event_time)

                    if delay > 0: time.sleep(delay)

                response = requests.post(
                    f"{SERVER_URL}/events",
                    json={
                        "event" : event,
                        "customer" : self.CUSTOMER,
                        "is_live" : True
                    }
                )
                response.raise_for_status()
                
                print(f"[EVENT] {event['event_id']} | {event['ingestion_time']} | {event['event_type']}")

                previous_event_time = current_event_time