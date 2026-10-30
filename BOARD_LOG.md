# BOARD_LOG

Ranks are appended after every check. Numbers come from judge logs only.

## 2026-10-30 03:30 UTC (Oct 30, 00:30 Brasilia)
Human submitted the four sub_pub1 files, all accepted. Overall rank 116.
| problem | file | board score |
|---|---|---|
| muse_attention_decode | sub_pub1.py | 2.580x |
| muse_decode_layer | sub_pub1.py | 2.136x |
| muse_moe_w4a16 | sub_pub1.py | 2.339x |
| muse_bf16_proj | sub_pub1.py | 1.500x |
Goal: overall top 50 (serving-queue priority), about +2-3% per problem.

## 2026-10-30 04:31 UTC: after the four public-ref checks
Leaderboard (okc leaderboard): overall rank 127 (4 problems). Earlier today the human reported 116; the board moved, our scores did not.
Checks run this round: 4 (attn Madhavan 2.600x, proj Madhavan 1.475x, layer vjeux 2.101x, moe vjeux 2.346x). None beats the sub_pub1 path; no new submission candidate.

## 2026-10-30 04:58 UTC: MoE port round (serving prize work, no board submission)
Five off-board checks on muse_moe_w4a16 (v3/v4 hot kernel, v5 pipe mix, v6 gather removal, clean sub_mk4). Board entries unchanged: attention 2.580x, decode_layer 2.136x, moe 2.339x, bf16_proj 1.500x (sub_pub1 files). Overall rank read after the round: 129 (was 127; others moved, our entries did not).
