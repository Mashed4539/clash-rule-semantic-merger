from __future__ import annotations
import os
import re
import yaml
import subprocess
import tempfile
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Iterable


class Config:
    MIHOMO_PATH = "mihomo"
    TEMP_DIR = "./temp"
    CONNECT_TIMEOUT = 10
    READ_TIMEOUT = 60
    MIHOMO_TIMEOUT = 120
    ENCODING = "utf-8"
    BUFFER_SIZE = 8192


@dataclass(frozen=True)
class Pattern:
    tokens: Tuple[str, ...]
    has_plus: bool
    prefix_len: int
    prefix: Tuple[str, ...]
    wildcard_count: int
    index: int


def _mask(tokens: Sequence[str]) -> int:
    mask = 0
    for i, tok in enumerate(tokens):
        if tok == "*":
            mask |= 1 << i
    return mask


class MaskIndex:
    def __init__(self, length: int) -> None:
        self.length = length
        self.by_mask: Dict[int, set[Tuple[str, ...]]] = {}
        self._pos_cache: Dict[int, Tuple[int, ...]] = {}
        self._mask_ids: Dict[int, int] = {}
        self._masks: List[int] = []
        self._bitset_by_pos: List[int] = [0] * length
        self._all_masks_bitset: int = 0

    def add(self, tokens: Sequence[str]) -> None:
        mask = _mask(tokens)
        mask_id = self._mask_ids.get(mask)
        if mask_id is None:
            mask_id = len(self._masks)
            self._mask_ids[mask] = mask_id
            self._masks.append(mask)
            self._all_masks_bitset |= 1 << mask_id
            for i in range(self.length):
                if (mask >> i) & 1:
                    self._bitset_by_pos[i] |= 1 << mask_id
        key = self._make_key(tokens, mask)
        bucket = self.by_mask.setdefault(mask, set())
        bucket.add(key)

    def _positions(self, mask: int) -> Tuple[int, ...]:
        pos = self._pos_cache.get(mask)
        if pos is None:
            pos = tuple(i for i in range(self.length) if not (mask >> i) & 1)
            self._pos_cache[mask] = pos
        return pos

    def _make_key(self, tokens: Sequence[str], mask: int) -> Tuple[str, ...]:
        return tuple(tokens[i] for i in self._positions(mask))

    def covers(self, tokens: Sequence[str]) -> bool:
        length = self.length
        if length == 0:
            return () in self.by_mask.get(0, set())

        q_mask = 0
        for i in range(length):
            if tokens[i] == "*":
                q_mask |= 1 << i

        by_mask = self.by_mask
        if not by_mask:
            return False

        lit_count = length - q_mask.bit_count()
        subset_count = 1 << lit_count
        mask_count = len(by_mask)

        if subset_count <= mask_count:
            lit_mask = ((1 << length) - 1) ^ q_mask
            sub = lit_mask
            while True:
                mask = q_mask | sub
                bucket = by_mask.get(mask)
                if bucket:
                    key = self._make_key(tokens, mask)
                    if key in bucket:
                        return True
                if sub == 0:
                    break
                sub = (sub - 1) & lit_mask
            return False

        if mask_count >= 64 and length > 0:
            candidates = self._all_masks_bitset
            qm = q_mask
            while qm:
                lsb = qm & -qm
                pos = lsb.bit_length() - 1
                candidates &= self._bitset_by_pos[pos]
                if candidates == 0:
                    return False
                qm ^= lsb
            while candidates:
                lsb = candidates & -candidates
                idx = lsb.bit_length() - 1
                mask = self._masks[idx]
                bucket = by_mask.get(mask)
                if bucket:
                    key = self._make_key(tokens, mask)
                    if key in bucket:
                        return True
                candidates ^= lsb
            return False

        for mask, bucket in by_mask.items():
            if (mask & q_mask) != q_mask:
                continue
            key = self._make_key(tokens, mask)
            if key in bucket:
                return True
        return False


def _parse_rule(rule: str, index: int) -> Pattern:
    parts = rule.split(".")
    if parts[0] == "":
        parts = ["+", "*"] + parts[1:]
    reversed_parts = parts[::-1]

    has_plus = reversed_parts[-1] == "+"
    if has_plus:
        prefix = tuple(reversed_parts[:-1])
    else:
        prefix = tuple(reversed_parts)
    wildcard_count = _mask(prefix).bit_count()
    return Pattern(
        tokens=tuple(reversed_parts),
        has_plus=has_plus,
        prefix_len=len(prefix),
        prefix=prefix,
        wildcard_count=wildcard_count,
        index=index,
    )


def minimize_rules(rules: Iterable[str]) -> List[str]:
    parsed: List[Pattern] = []
    for idx, rule in enumerate(rules):
        parsed.append(_parse_rule(rule, idx))

    def sort_key(p: Pattern) -> Tuple[int, int, int, int]:
        return (0 if p.has_plus else 1, p.prefix_len, -p.wildcard_count, p.index)

    ordered = sorted(parsed, key=sort_key)
    keep = [False] * len(parsed)

    fixed_indices: Dict[int, MaskIndex] = {}
    plus_indices: Dict[int, MaskIndex] = {}
    plus_lengths: List[int] = []

    def _ensure_plus_index(length: int) -> MaskIndex:
        idx = plus_indices.get(length)
        if idx is None:
            idx = MaskIndex(length)
            plus_indices[length] = idx
            insert_pos = 0
            while insert_pos < len(plus_lengths) and plus_lengths[insert_pos] < length:
                insert_pos += 1
            plus_lengths.insert(insert_pos, length)
        return idx

    for p in ordered:
        covered = False

        if p.has_plus:
            for length in plus_lengths:
                if length > p.prefix_len:
                    break
                if plus_indices[length].covers(p.prefix[:length]):
                    covered = True
                    break
        else:
            for length in plus_lengths:
                if length > p.prefix_len:
                    break
                if plus_indices[length].covers(p.prefix[:length]):
                    covered = True
                    break
            if not covered:
                fixed_idx = fixed_indices.get(p.prefix_len)
                if fixed_idx and fixed_idx.covers(p.prefix):
                    covered = True

        if covered:
            continue

        keep[p.index] = True
        if p.has_plus:
            _ensure_plus_index(p.prefix_len).add(p.prefix)
        else:
            fixed_idx = fixed_indices.get(p.prefix_len)
            if fixed_idx is None:
                fixed_idx = MaskIndex(p.prefix_len)
                fixed_indices[p.prefix_len] = fixed_idx
            fixed_idx.add(p.prefix)

    result = []
    for i in range(len(parsed)):
        if keep[i]:
            p = parsed[i]
            reversed_back = p.tokens[::-1]
            rule_str = ".".join(reversed_back)
            result.append(rule_str)
    return result


class RuleExtractor:
    __slots__ = ("mihomo", "session")

    DOMAIN_PATTERN = re.compile(
        r"^(?:\+\.)?\.?(?:(?:\*|xn--[a-zA-Z0-9]+|[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)\.)*(?:\*|xn--[a-zA-Z0-9]+|[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)$"
    )
    IPV4_CIDR_PATTERN = re.compile(
        r"^((?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d))/(3[0-2]|[12]?\d)$"
    )

    def __init__(self, mihomo):
        self.mihomo = mihomo
        self.session = requests.Session()

    def extract(self, source_config: Dict, behavior: str) -> Tuple[List[str], int]:
        source_type = source_config["type"]
        content = self._fetch_content(source_config, source_type)
        raw_rules = self._parse_content(
            content, source_config.get("format", "text"), behavior
        )
        cleaned_rules = self._clean_and_validate(raw_rules, behavior)
        return cleaned_rules, len(raw_rules)

    def _fetch_content(self, source_config: Dict, source_type: str) -> bytes:
        if source_type == "http":
            try:
                resp = self.session.get(
                    source_config["url"],
                    timeout=(Config.CONNECT_TIMEOUT, Config.READ_TIMEOUT),
                    stream=True,
                )
                resp.raise_for_status()
                return resp.content
            except Exception:
                return b""
        elif source_type == "file":
            with open(source_config["path"], "rb", buffering=Config.BUFFER_SIZE) as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _parse_content(
        self, content: bytes, source_format: str, behavior: str
    ) -> List[str]:
        try:
            if source_format == "mrs":
                return self._parse_mrs(content, behavior)
            text_content = content.decode(Config.ENCODING, errors="ignore")
            return (
                self._parse_yaml(text_content)
                if source_format == "yaml"
                else self._clean_lines(text_content.splitlines())
            )
        except Exception:
            return []

    def _parse_yaml(self, text_content: str) -> List[str]:
        data = yaml.safe_load(text_content)
        if isinstance(data, dict) and "payload" in data:
            rules = data["payload"]
        elif isinstance(data, list):
            rules = data
        else:
            rules = text_content.splitlines()
        return self._clean_lines(rules)

    def _parse_mrs(self, content: bytes, behavior: str) -> List[str]:
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=Config.TEMP_DIR, suffix=".mrs", delete=True
        ) as mrs_file:
            mrs_file.write(content)
            mrs_file.flush()
            with tempfile.NamedTemporaryFile(
                mode="r+", dir=Config.TEMP_DIR, suffix=".txt", delete=True
            ) as txt_file:
                self.mihomo.convert_ruleset(
                    behavior, "mrs", mrs_file.name, txt_file.name
                )
                txt_file.seek(0)
                return self._clean_lines(txt_file.read().splitlines())

    def _clean_lines(self, lines: List[str]) -> List[str]:
        res = []
        for line in lines:
            s = line.strip().strip("'\"")
            if s and not s.startswith("#"):
                res.append(s)
        return res

    def _clean_and_validate(self, raw_rules: List[str], behavior: str) -> List[str]:
        if behavior == "domain":
            return [
                r for r in raw_rules if len(r) <= 255 and self.DOMAIN_PATTERN.match(r)
            ]
        elif behavior == "ipcidr":
            return [r for r in raw_rules if self.IPV4_CIDR_PATTERN.match(r)]
        return raw_rules


class MihomoClient:
    __slots__ = ("binary_path",)

    def __init__(self, binary_path: str = Config.MIHOMO_PATH):
        self.binary_path = binary_path

    def convert_ruleset(
        self, behavior: str, src_fmt: str, src_path: str, dst_path: str
    ) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
            os.makedirs(Config.TEMP_DIR, exist_ok=True)
            subprocess.run(
                [
                    self.binary_path,
                    "convert-ruleset",
                    behavior,
                    src_fmt,
                    src_path,
                    dst_path,
                ],
                capture_output=True,
                text=True,
                bufsize=Config.BUFFER_SIZE,
                timeout=Config.MIHOMO_TIMEOUT,
            )
        except Exception:
            pass


class RuleOptimizer:
    __slots__ = ("mihomo",)

    def __init__(self, mihomo):
        self.mihomo = mihomo

    def optimize_domains(self, rules: List[str]) -> List[str]:
        return minimize_rules(rules)

    def optimize_ipcidr(self, rules: List[str]) -> List[str]:
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=Config.TEMP_DIR, suffix=".txt", delete=True
        ) as txt_file:
            txt_file.write("\n".join(rules))
            txt_file.flush()
            with tempfile.NamedTemporaryFile(
                dir=Config.TEMP_DIR, suffix=".mrs", delete=True
            ) as mrs_file:
                self.mihomo.convert_ruleset(
                    "ipcidr", "text", txt_file.name, mrs_file.name
                )
                with tempfile.NamedTemporaryFile(
                    mode="r+", dir=Config.TEMP_DIR, suffix=".txt", delete=True
                ) as out_file:
                    self.mihomo.convert_ruleset(
                        "ipcidr", "mrs", mrs_file.name, out_file.name
                    )
                    out_file.seek(0)
                    return [line.strip() for line in out_file if line.strip()]


class RuleWriter:
    __slots__ = ("mihomo",)

    def __init__(self, mihomo):
        self.mihomo = mihomo

    def write(
        self, path: str, rules: List[str], output_format: str, behavior: str
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        os.makedirs(Config.TEMP_DIR, exist_ok=True)

        if output_format == "mrs":
            self._write_mrs(path, rules, behavior)
        elif output_format == "yaml":
            self._write_yaml(path, rules)
        else:
            self._write_text(path, rules)

    def _write_mrs(self, path: str, rules: List[str], behavior: str) -> None:
        if not rules:
            open(path, "w").close()
            return
        with tempfile.NamedTemporaryFile(
            mode="w", dir=Config.TEMP_DIR, suffix=".txt", delete=True
        ) as txt_file:
            txt_file.write("\n".join(rules))
            txt_file.flush()
            self.mihomo.convert_ruleset(behavior, "text", txt_file.name, path)

    def _write_yaml(self, path: str, rules: List[str]) -> None:
        content = f"# Updated: {datetime.now()}\npayload:\n" + "\n".join(
            f"  - '{r}'" for r in rules
        )
        with open(
            path, "w", encoding=Config.ENCODING, buffering=Config.BUFFER_SIZE
        ) as f:
            f.write(content)

    def _write_text(self, path: str, rules: List[str]) -> None:
        with open(
            path, "w", encoding=Config.ENCODING, buffering=Config.BUFFER_SIZE
        ) as f:
            f.write("\n".join(rules))


@dataclass
class JobConfig:
    path: str
    behavior: str
    format: str
    upstream: Dict[str, Dict]


class RuleMergeJob:
    __slots__ = ("config", "mihomo", "extractor", "optimizer", "writer")

    def __init__(self, config: JobConfig):
        self.config = config
        self.mihomo = MihomoClient()
        self.extractor = RuleExtractor(self.mihomo)
        self.optimizer = RuleOptimizer(self.mihomo)
        self.writer = RuleWriter(self.mihomo)

    def run(self) -> None:
        print(
            f"任务{self.config.path} 类型:{self.config.behavior}({self.config.format})",
            flush=True,
        )

        all_rules = []
        for name, source in self.config.upstream.items():
            cleaned, raw_cnt = self.extractor.extract(source, self.config.behavior)
            print(f"    上游{name},合法规则{len(cleaned)}/{raw_cnt}个", flush=True)
            all_rules.extend(cleaned)

        final_rules = self._optimize_rules(all_rules)
        print(f"语义最简规则{len(final_rules)}/{len(all_rules)}个\n", flush=True)
        self._write_output(final_rules)

    def _optimize_rules(self, rules: List[str]) -> List[str]:
        if self.config.behavior == "domain":
            return self.optimizer.optimize_domains(rules)
        elif self.config.behavior == "ipcidr":
            return self.optimizer.optimize_ipcidr(rules)
        else:
            return []

    def _write_output(self, rules: List[str]) -> None:
        self.writer.write(
            self.config.path, rules, self.config.format, self.config.behavior
        )


class RulesMerger:
    __slots__ = ("config_path",)

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def run(self) -> None:
        with open(
            self.config_path,
            "r",
            encoding=Config.ENCODING,
            buffering=Config.BUFFER_SIZE,
        ) as f:
            config_list = yaml.safe_load(f)

        for config_dict in config_list:
            job_config = JobConfig(
                path=os.path.join("output", config_dict["path"]),
                behavior=config_dict["behavior"],
                format=config_dict.get("format", "yaml"),
                upstream=config_dict["upstream"],
            )
            RuleMergeJob(job_config).run()


if __name__ == "__main__":
    RulesMerger().run()
