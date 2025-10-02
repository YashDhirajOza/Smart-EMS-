from __future__ import annotations

from typing import Any, Dict


def inverter_overheat_resolver(metric_value: float) -> Dict[str, Any]:
    return {
        "action": "Reduce inverter load by 20% and increase ventilation.",
        "metric_value": metric_value,
        "priority": "high",
    }


def battery_soc_low_resolver(metric_value: float) -> Dict[str, Any]:
    return {
        "action": "Initiate grid charging sequence and defer EV charging.",
        "metric_value": metric_value,
        "priority": "medium",
    }


def ev_charger_overload_resolver(metric_value: float) -> Dict[str, Any]:
    return {
        "action": "Throttle EV charging rate to 50% for next 10 minutes.",
        "metric_value": metric_value,
        "priority": "critical" if metric_value > 32 else "high",
    }


RESOLVER_REGISTRY = {
    "inverter_temperature": inverter_overheat_resolver,
    "state_of_charge": battery_soc_low_resolver,
    "charger_current": ev_charger_overload_resolver,
}
