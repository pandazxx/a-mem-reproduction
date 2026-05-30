# A-Mem Retrieval Traces

Solid arrows = direct vector-search hits.  
Dashed arrows = one-hop A-Mem link traversals.

## q01 — single_hop

**Q:** Where does Sam live?  
**Expected:** Oakland, California  
**Got:** Sam lives in Oakland, California.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Where does Sam live?"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m35["m35: Alex will join Sam on the Boston trip."]
    Q ==> m35
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    m11 -.->|link| m05
    m34["m34: Sam booked flights to Boston for May 13…"]
    m35 -.->|link| m34
```

## q02 — single_hop

**Q:** What is Maria's profession?  
**Expected:** high school chemistry teacher  
**Got:** Maria works as a high school chemistry teacher.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Maria's profession?"])
    m12["m12: Maria works as a high school chemistry…"]
    Q ==> m12
    m33["m33: Maria's birthday is May 15."]
    Q ==> m33
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m15["m15: Sam plans to visit Maria for her…"]
    Q ==> m15
    m36["m36: David will also fly to Boston for…"]
    Q ==> m36
    m04["m04: Sam lives in Oakland, California."]
    m11 -.->|link| m04
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    m11 -.->|link| m05
    m34["m34: Sam booked flights to Boston for May 13…"]
    m36 -.->|link| m34
```

## q03 — single_hop

**Q:** When is Maria's birthday?  
**Expected:** May 15  
**Got:** Maria's birthday is May 15.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: When is Maria's birthday?"])
    m33["m33: Maria's birthday is May 15."]
    Q ==> m33
    m15["m15: Sam plans to visit Maria for her…"]
    Q ==> m15
    m36["m36: David will also fly to Boston for…"]
    Q ==> m36
    m34["m34: Sam booked flights to Boston for May 13…"]
    Q ==> m34
    m12["m12: Maria works as a high school chemistry…"]
    Q ==> m12
    m11["m11: Sam's mother Maria lives in Boston."]
    m15 -.->|link| m11
    m04["m04: Sam lives in Oakland, California."]
    m15 -.->|link| m04
```

## q04 — single_hop

**Q:** What climbing shoes did Sam order?  
**Expected:** La Sportiva Solution  
**Got:** Sam ordered the La Sportiva Solution climbing shoes.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What climbing shoes did Sam order?"])
    m39["m39: Sam ordered a new pair of climbing…"]
    Q ==> m39
    m09["m09: Sam has been rock climbing for three…"]
    Q ==> m09
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m21["m21: Sam had an amazing climbing trip to…"]
    Q ==> m21
    m10["m10: Sam's favorite climbing problems are…"]
    Q ==> m10
    m38["m38: Sam went climbing at Berkeley Ironworks…"]
    m39 -.->|link| m38
    m04["m04: Sam lives in Oakland, California."]
    m08 -.->|link| m04
    m20["m20: Sam's goal with the scraper is to…"]
    m21 -.->|link| m20
```

## q05 — two_hop

**Q:** In which city is the company that David works for headquartered?  
**Expected:** Seattle  
**Got:** The company that David works for, Amazon, is headquartered in Seattle.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: In which city is the company that David works for…"])
    m14["m14: David works at Amazon as a product…"]
    Q ==> m14
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m36["m36: David will also fly to Boston for…"]
    Q ==> m36
    m26["m26: StartupCo is based in San Francisco but…"]
    Q ==> m26
    m02["m02: TechCorp is a fintech company…"]
    Q ==> m02
    m04["m04: Sam lives in Oakland, California."]
    m13 -.->|link| m04
    m11["m11: Sam's mother Maria lives in Boston."]
    m13 -.->|link| m11
    m34["m34: Sam booked flights to Boston for May 13…"]
    m36 -.->|link| m34
    m25["m25: Sam got a job offer from StartupCo, a…"]
    m26 -.->|link| m25
    m01["m01: Sam works as a software engineer at…"]
    m02 -.->|link| m01
```

## q06 — two_hop

**Q:** What programming language does Sam's web scraper use?  
**Expected:** Python (initially), then rewritten in Rust  
**Got:** Sam's web scraper uses Rust.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What programming language does Sam's web scraper use?"])
    m17["m17: Sam decided to use Python for the web…"]
    Q ==> m17
    m18["m18: Sam uses the BeautifulSoup library for…"]
    Q ==> m18
    m16["m16: Sam started a side project: building a…"]
    Q ==> m16
    m32["m32: Sam is using the reqwest and scraper…"]
    Q ==> m32
    m31["m31: Sam decided to rewrite the climbing…"]
    Q ==> m31
    m09["m09: Sam has been rock climbing for three…"]
    m16 -.->|link| m09
    m08["m08: Sam goes rock climbing at Berkeley…"]
    m16 -.->|link| m08
    m20["m20: Sam's goal with the scraper is to…"]
    m31 -.->|link| m20
```

## q07 — two_hop

**Q:** What does Sam's mother do for a living, and where does she live?  
**Expected:** high school chemistry teacher in Boston  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What does Sam's mother do for a living, and where does she…"])
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
```

## q08 — deep_multi_hop

**Q:** Who is the manager of the person whose brother works at Amazon?  
**Expected:** Marcus  
**Got:** Jennifer is the manager of the person whose brother works at Amazon.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: Who is the manager of the person whose brother works at…"])
    m14["m14: David works at Amazon as a product…"]
    Q ==> m14
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    Q ==> m03
    m04["m04: Sam lives in Oakland, California."]
    m13 -.->|link| m04
    m11["m11: Sam's mother Maria lives in Boston."]
    m13 -.->|link| m11
```

## q09 — deep_multi_hop

**Q:** What is the profession of the partner of the engineer who has a brother in Seattle?  
**Expected:** graphic designer  
**Got:** The partner of the engineer is a graphic designer.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is the profession of the partner of the engineer who…"])
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m14["m14: David works at Amazon as a product…"]
    Q ==> m14
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m29["m29: Sam started at StartupCo today. The…"]
    Q ==> m29
    m04["m04: Sam lives in Oakland, California."]
    m13 -.->|link| m04
    m11["m11: Sam's mother Maria lives in Boston."]
    m13 -.->|link| m11
    m25["m25: Sam got a job offer from StartupCo, a…"]
    m29 -.->|link| m25
    m24["m24: Sam started interviewing at other…"]
    m29 -.->|link| m24
```

## q10 — deep_multi_hop

**Q:** In which city was the person born who is married to the graphic designer's partner's brother?  
**Expected:** unknown / cannot determine (Maria's birth location is not in the facts)  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: In which city was the person born who is married to the…"])
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    Q ==> m03
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m04["m04: Sam lives in Oakland, California."]
    m13 -.->|link| m04
```

## q11 — implicit_conceptual

**Q:** What outdoor activities does Sam enjoy?  
**Expected:** rock climbing (Yosemite, Joshua Tree, Berkeley Ironworks — though gym is indoor)  
**Got:** Sam enjoys rock climbing.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What outdoor activities does Sam enjoy?"])
    m21["m21: Sam had an amazing climbing trip to…"]
    Q ==> m21
    m09["m09: Sam has been rock climbing for three…"]
    Q ==> m09
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m20["m20: Sam's goal with the scraper is to…"]
    Q ==> m20
    m16["m16: Sam started a side project: building a…"]
    m20 -.->|link| m16
```

## q12 — implicit_conceptual

**Q:** What Python libraries has Sam used?  
**Expected:** BeautifulSoup  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What Python libraries has Sam used?"])
    m17["m17: Sam decided to use Python for the web…"]
    Q ==> m17
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m07["m07: Sam uses Duolingo daily for Spanish…"]
    Q ==> m07
    m24["m24: Sam started interviewing at other…"]
    Q ==> m24
    m25["m25: Sam got a job offer from StartupCo, a…"]
    Q ==> m25
    m16["m16: Sam started a side project: building a…"]
    m17 -.->|link| m16
    m06["m06: Sam started learning Spanish two months…"]
    m07 -.->|link| m06
```

## q13 — implicit_conceptual

**Q:** What hobbies has Sam been less active in lately?  
**Expected:** Spanish learning (slipped to once a week) and climbing (first session in two weeks before m38)  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What hobbies has Sam been less active in lately?"])
    m09["m09: Sam has been rock climbing for three…"]
    Q ==> m09
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m21["m21: Sam had an amazing climbing trip to…"]
    Q ==> m21
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m24["m24: Sam started interviewing at other…"]
    Q ==> m24
    m20["m20: Sam's goal with the scraper is to…"]
    m21 -.->|link| m20
    m01["m01: Sam works as a software engineer at…"]
    m24 -.->|link| m01
```

## q14 — information_update

**Q:** Where does Sam currently work?  
**Expected:** StartupCo  
**Got:** Sam currently works at StartupCo.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Where does Sam currently work?"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m24["m24: Sam started interviewing at other…"]
    Q ==> m24
    m29["m29: Sam started at StartupCo today. The…"]
    Q ==> m29
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m25["m25: Sam got a job offer from StartupCo, a…"]
    m29 -.->|link| m25
```

## q15 — information_update

**Q:** Who is Sam's current manager?  
**Expected:** Marcus  
**Got:** Sam's current manager is Marcus.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Who is Sam's current manager?"])
    m28["m28: Sam's new manager at StartupCo will be…"]
    Q ==> m28
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    Q ==> m03
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m24["m24: Sam started interviewing at other…"]
    Q ==> m24
    m25["m25: Sam got a job offer from StartupCo, a…"]
    m28 -.->|link| m25
```

## q16 — information_update

**Q:** What language is Sam's climbing scraper currently written in?  
**Expected:** Rust  
**Got:** Sam's climbing scraper is currently written in Rust.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What language is Sam's climbing scraper currently written…"])
    m31["m31: Sam decided to rewrite the climbing…"]
    Q ==> m31
    m20["m20: Sam's goal with the scraper is to…"]
    Q ==> m20
    m16["m16: Sam started a side project: building a…"]
    Q ==> m16
    m09["m09: Sam has been rock climbing for three…"]
    Q ==> m09
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m04["m04: Sam lives in Oakland, California."]
    m08 -.->|link| m04
```

## q17 — compositional_aggregation

**Q:** Who are all the people Sam has mentioned in conversations?  
**Expected:** Alex (partner), Jennifer (old manager), Maria (mother), David (brother), Marcus (new manager)  
**Got:** Sam has mentioned Alex, Maria, Jennifer, and Marcus in conversations.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: Who are all the people Sam has mentioned in conversations?"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m24["m24: Sam started interviewing at other…"]
    Q ==> m24
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m28["m28: Sam's new manager at StartupCo will be…"]
    Q ==> m28
    m01["m01: Sam works as a software engineer at…"]
    m24 -.->|link| m01
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
    m25["m25: Sam got a job offer from StartupCo, a…"]
    m28 -.->|link| m25
```

## q18 — compositional_aggregation

**Q:** What different programming languages and libraries has Sam used or learned?  
**Expected:** Python, BeautifulSoup, Rust, reqwest, scraper  
**Got:** Sam has used or learned Python, Rust, and possibly others for work, but the specific libraries are unknown / not mentioned.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What different programming languages and libraries has Sam…"])
    m06["m06: Sam started learning Spanish two months…"]
    Q ==> m06
    m07["m07: Sam uses Duolingo daily for Spanish…"]
    Q ==> m07
    m30["m30: Sam is now learning Rust because…"]
    Q ==> m30
    m17["m17: Sam decided to use Python for the web…"]
    Q ==> m17
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m04["m04: Sam lives in Oakland, California."]
    m06 -.->|link| m04
    m29["m29: Sam started at StartupCo today. The…"]
    m30 -.->|link| m29
    m16["m16: Sam started a side project: building a…"]
    m17 -.->|link| m16
```

## q19 — compositional_aggregation

**Q:** What places does Sam have travel or residence connections to?  
**Expected:** Oakland (lives), San Francisco (work), Boston (mother + planned trip), Seattle (brother), Yosemite (climbed), Joshua Tree (planned trip), Mexico (Spanish learning motivation)  
**Got:** Sam has connections to Oakland, Boston, and Seattle.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What places does Sam have travel or residence connections…"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m35["m35: Alex will join Sam on the Boston trip."]
    Q ==> m35
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    m11 -.->|link| m05
    m34["m34: Sam booked flights to Boston for May 13…"]
    m35 -.->|link| m34
```

## q20 — absence_abstention

**Q:** What is Sam's favorite food?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Sam's favorite food?"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m21["m21: Sam had an amazing climbing trip to…"]
    Q ==> m21
    m35["m35: Alex will join Sam on the Boston trip."]
    Q ==> m35
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m08["m08: Sam goes rock climbing at Berkeley…"]
    Q ==> m08
    m09["m09: Sam has been rock climbing for three…"]
    m21 -.->|link| m09
    m20["m20: Sam's goal with the scraper is to…"]
    m21 -.->|link| m20
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    m35 -.->|link| m05
    m34["m34: Sam booked flights to Boston for May 13…"]
    m35 -.->|link| m34
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
```

## q21 — absence_abstention

**Q:** Does Sam have any pets?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Does Sam have any pets?"])
    m04["m04: Sam lives in Oakland, California."]
    Q ==> m04
    m35["m35: Alex will join Sam on the Boston trip."]
    Q ==> m35
    m13["m13: Sam's brother David lives in Seattle."]
    Q ==> m13
    m11["m11: Sam's mother Maria lives in Boston."]
    Q ==> m11
    m05["m05: Sam's partner is Alex. Alex works as a…"]
    Q ==> m05
    m34["m34: Sam booked flights to Boston for May 13…"]
    m35 -.->|link| m34
    m03["m03: Sam's manager at TechCorp is Jennifer."]
    m11 -.->|link| m03
    m01["m01: Sam works as a software engineer at…"]
    m05 -.->|link| m01
```

## q22 — absence_abstention

**Q:** What is Sam's salary at StartupCo?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Sam's salary at StartupCo?"])
    m29["m29: Sam started at StartupCo today. The…"]
    Q ==> m29
    m25["m25: Sam got a job offer from StartupCo, a…"]
    Q ==> m25
    m27["m27: Sam accepted the StartupCo offer and…"]
    Q ==> m27
    m28["m28: Sam's new manager at StartupCo will be…"]
    Q ==> m28
    m01["m01: Sam works as a software engineer at…"]
    Q ==> m01
    m24["m24: Sam started interviewing at other…"]
    m29 -.->|link| m24
```
