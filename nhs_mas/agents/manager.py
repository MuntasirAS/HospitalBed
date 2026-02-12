
class NHSMASManager:
    def __init__(self, agents, model, scaler, mas_cfg):
        self.agents = agents
        self.model = model
        self.scaler = scaler
        self.cfg = mas_cfg

    def balance(self, sector: str):
        requesters = []
        # 1) Forecast phase
        for a in self.agents:
            at_risk, pred = a.forecast_rate_risk(
                sector, self.model, self.scaler,
                lookback=4,
                threshold=self.cfg.risk_threshold
            )
            if at_risk:
                requesters.append((a, pred))

        # 2) Negotiation phase
        for req, pred in requesters:
            providers = [a for a in self.agents
                         if (a is not req) and (a.get_occ_rate(sector) < self.cfg.provider_capacity_threshold)]

            if not providers:
                # Inter-regional same logic already covered since we don't filter by region here; extend if needed
                print(f"[WARNING] No capacity for {req.org_code} in {sector}")
                continue

            # choose lowest occupancy provider
            best = min(providers, key=lambda x: x.get_occ_rate(sector))

            # dynamic transfer fraction of current occ
            occ = req.sectors[sector]['occ']
            transfer = occ * self.cfg.transfer_percentage
            # capacity cap
            cap_left = best.sectors[sector]['avail'] - best.sectors[sector]['occ']
            actual = max(0.0, min(transfer, cap_left))
            if actual <= 0:
                continue

            req.sectors[sector]['occ'] -= actual
            best.sectors[sector]['occ'] += actual

            print(f"[ACTION] {req.org_code} (pred {pred:.1%}) -> {best.org_code} [{sector}] x{actual:.2f}")
