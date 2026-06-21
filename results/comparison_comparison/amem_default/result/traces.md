# A-Mem Retrieval Traces

Solid arrows = direct vector-search hits.  
Dashed arrows = one-hop A-Mem link traversals.

## Q00001 — single_hop

**Q:** Where does Sam live?  
**Expected:** Oakland, California  
**Got:** Sam lives in Oakland, California.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Where does Sam live?"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00035["S00035: Alex will join Sam on the Boston trip."]
    Q ==> S00035
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00001["S00001: Sam works as a software engineer at…"]
    S00011 -.->|link| S00001
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    S00035 -.->|link| S00005
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00035 -.->|link| S00034
```

## Q00002 — single_hop

**Q:** What is Maria's profession?  
**Expected:** high school chemistry teacher  
**Got:** Maria works as a high school chemistry teacher.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Maria's profession?"])
    S00012["S00012: Maria works as a high school chemistry…"]
    Q ==> S00012
    S00033["S00033: Maria's birthday is May 15."]
    Q ==> S00033
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00015["S00015: Sam plans to visit Maria for her…"]
    Q ==> S00015
    S00036["S00036: David will also fly to Boston for…"]
    Q ==> S00036
    S00004["S00004: Sam lives in Oakland, California."]
    S00011 -.->|link| S00004
    S00001["S00001: Sam works as a software engineer at…"]
    S00011 -.->|link| S00001
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00036 -.->|link| S00034
```

## Q00003 — single_hop

**Q:** When is Maria's birthday?  
**Expected:** May 15  
**Got:** Maria's birthday is May 15.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: When is Maria's birthday?"])
    S00033["S00033: Maria's birthday is May 15."]
    Q ==> S00033
    S00015["S00015: Sam plans to visit Maria for her…"]
    Q ==> S00015
    S00036["S00036: David will also fly to Boston for…"]
    Q ==> S00036
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    Q ==> S00034
    S00012["S00012: Maria works as a high school chemistry…"]
    Q ==> S00012
    S00011["S00011: Sam's mother Maria lives in Boston."]
    S00015 -.->|link| S00011
```

## Q00004 — single_hop

**Q:** What climbing shoes did Sam order?  
**Expected:** La Sportiva Solution  
**Got:** Sam ordered the La Sportiva Solution climbing shoes.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What climbing shoes did Sam order?"])
    S00039["S00039: Sam ordered a new pair of climbing…"]
    Q ==> S00039
    S00009["S00009: Sam has been rock climbing for three…"]
    Q ==> S00009
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00021["S00021: Sam had an amazing climbing trip to…"]
    Q ==> S00021
    S00010["S00010: Sam's favorite climbing problems are…"]
    Q ==> S00010
    S00038["S00038: Sam went climbing at Berkeley Ironworks…"]
    S00039 -.->|link| S00038
    S00004["S00004: Sam lives in Oakland, California."]
    S00008 -.->|link| S00004
    S00001["S00001: Sam works as a software engineer at…"]
    S00008 -.->|link| S00001
    S00020["S00020: Sam's goal with the scraper is to…"]
    S00021 -.->|link| S00020
```

## Q00005 — two_hop

**Q:** In which city is the company that David works for headquartered?  
**Expected:** Seattle  
**Got:** The company that David works for, Amazon, is headquartered in Seattle.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: In which city is the company that David works for…"])
    S00014["S00014: David works at Amazon as a product…"]
    Q ==> S00014
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00036["S00036: David will also fly to Boston for…"]
    Q ==> S00036
    S00026["S00026: StartupCo is based in San Francisco but…"]
    Q ==> S00026
    S00002["S00002: TechCorp is a fintech company…"]
    Q ==> S00002
    S00011["S00011: Sam's mother Maria lives in Boston."]
    S00013 -.->|link| S00011
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00036 -.->|link| S00034
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00026 -.->|link| S00025
    S00001["S00001: Sam works as a software engineer at…"]
    S00002 -.->|link| S00001
```

## Q00006 — two_hop

**Q:** What programming language does Sam's web scraper use?  
**Expected:** Python (initially), then rewritten in Rust  
**Got:** Sam's web scraper uses Rust.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What programming language does Sam's web scraper use?"])
    S00017["S00017: Sam decided to use Python for the web…"]
    Q ==> S00017
    S00018["S00018: Sam uses the BeautifulSoup library for…"]
    Q ==> S00018
    S00016["S00016: Sam started a side project: building a…"]
    Q ==> S00016
    S00032["S00032: Sam is using the reqwest and scraper…"]
    Q ==> S00032
    S00031["S00031: Sam decided to rewrite the climbing…"]
    Q ==> S00031
    S00001["S00001: Sam works as a software engineer at…"]
    S00018 -.->|link| S00001
    S00009["S00009: Sam has been rock climbing for three…"]
    S00016 -.->|link| S00009
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    S00016 -.->|link| S00008
    S00020["S00020: Sam's goal with the scraper is to…"]
    S00031 -.->|link| S00020
```

## Q00007 — two_hop

**Q:** What does Sam's mother do for a living, and where does she live?  
**Expected:** high school chemistry teacher in Boston  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What does Sam's mother do for a living, and where does she…"])
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
```

## Q00008 — deep_multi_hop

**Q:** Who is the manager of the person whose brother works at Amazon?  
**Expected:** Marcus  
**Got:** Jennifer is the manager of the person whose brother works at Amazon.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: Who is the manager of the person whose brother works at…"])
    S00014["S00014: David works at Amazon as a product…"]
    Q ==> S00014
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00003["S00003: Sam's manager at TechCorp is Jennifer."]
    Q ==> S00003
    S00011["S00011: Sam's mother Maria lives in Boston."]
    S00013 -.->|link| S00011
```

## Q00009 — deep_multi_hop

**Q:** What is the profession of the partner of the engineer who has a brother in Seattle?  
**Expected:** graphic designer  
**Got:** The partner of the engineer is a graphic designer.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is the profession of the partner of the engineer who…"])
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00014["S00014: David works at Amazon as a product…"]
    Q ==> S00014
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00029["S00029: Sam started at StartupCo today. The…"]
    Q ==> S00029
    S00011["S00011: Sam's mother Maria lives in Boston."]
    S00013 -.->|link| S00011
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00029 -.->|link| S00025
    S00024["S00024: Sam started interviewing at other…"]
    S00029 -.->|link| S00024
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    S00029 -.->|link| S00027
```

## Q00010 — deep_multi_hop

**Q:** In which city was the person born who is married to the graphic designer's partner's brother?  
**Expected:** unknown / cannot determine (Maria's birth location is not in the facts)  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: In which city was the person born who is married to the…"])
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00003["S00003: Sam's manager at TechCorp is Jennifer."]
    Q ==> S00003
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00004["S00004: Sam lives in Oakland, California."]
    S00011 -.->|link| S00004
```

## Q00011 — implicit_conceptual

**Q:** What outdoor activities does Sam enjoy?  
**Expected:** rock climbing (Yosemite, Joshua Tree, Berkeley Ironworks — though gym is indoor)  
**Got:** Sam enjoys rock climbing.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What outdoor activities does Sam enjoy?"])
    S00021["S00021: Sam had an amazing climbing trip to…"]
    Q ==> S00021
    S00009["S00009: Sam has been rock climbing for three…"]
    Q ==> S00009
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00020["S00020: Sam's goal with the scraper is to…"]
    Q ==> S00020
    S00001["S00001: Sam works as a software engineer at…"]
    S00008 -.->|link| S00001
    S00016["S00016: Sam started a side project: building a…"]
    S00020 -.->|link| S00016
```

## Q00012 — implicit_conceptual

**Q:** What Python libraries has Sam used?  
**Expected:** BeautifulSoup  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What Python libraries has Sam used?"])
    S00017["S00017: Sam decided to use Python for the web…"]
    Q ==> S00017
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00007["S00007: Sam uses Duolingo daily for Spanish…"]
    Q ==> S00007
    S00024["S00024: Sam started interviewing at other…"]
    Q ==> S00024
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    Q ==> S00025
    S00016["S00016: Sam started a side project: building a…"]
    S00017 -.->|link| S00016
    S00006["S00006: Sam started learning Spanish two months…"]
    S00007 -.->|link| S00006
```

## Q00013 — implicit_conceptual

**Q:** What hobbies has Sam been less active in lately?  
**Expected:** Spanish learning (slipped to once a week) and climbing (first session in two weeks before m38)  
**Got:** unknown / not mentioned  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What hobbies has Sam been less active in lately?"])
    S00009["S00009: Sam has been rock climbing for three…"]
    Q ==> S00009
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00021["S00021: Sam had an amazing climbing trip to…"]
    Q ==> S00021
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00024["S00024: Sam started interviewing at other…"]
    Q ==> S00024
    S00020["S00020: Sam's goal with the scraper is to…"]
    S00021 -.->|link| S00020
    S00001["S00001: Sam works as a software engineer at…"]
    S00008 -.->|link| S00001
```

## Q00014 — information_update

**Q:** Where does Sam currently work?  
**Expected:** StartupCo  
**Got:** Sam currently works at StartupCo.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Where does Sam currently work?"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00024["S00024: Sam started interviewing at other…"]
    Q ==> S00024
    S00029["S00029: Sam started at StartupCo today. The…"]
    Q ==> S00029
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00029 -.->|link| S00025
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    S00029 -.->|link| S00027
```

## Q00015 — information_update

**Q:** Who is Sam's current manager?  
**Expected:** Marcus  
**Got:** Sam's current manager is Marcus.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Who is Sam's current manager?"])
    S00028["S00028: Sam's new manager at StartupCo will be…"]
    Q ==> S00028
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00003["S00003: Sam's manager at TechCorp is Jennifer."]
    Q ==> S00003
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00024["S00024: Sam started interviewing at other…"]
    Q ==> S00024
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    S00028 -.->|link| S00027
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00028 -.->|link| S00025
```

## Q00016 — information_update

**Q:** What language is Sam's climbing scraper currently written in?  
**Expected:** Rust  
**Got:** Sam's climbing scraper is currently written in Rust.  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What language is Sam's climbing scraper currently written…"])
    S00031["S00031: Sam decided to rewrite the climbing…"]
    Q ==> S00031
    S00020["S00020: Sam's goal with the scraper is to…"]
    Q ==> S00020
    S00016["S00016: Sam started a side project: building a…"]
    Q ==> S00016
    S00009["S00009: Sam has been rock climbing for three…"]
    Q ==> S00009
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00004["S00004: Sam lives in Oakland, California."]
    S00008 -.->|link| S00004
    S00001["S00001: Sam works as a software engineer at…"]
    S00008 -.->|link| S00001
```

## Q00017 — compositional_aggregation

**Q:** Who are all the people Sam has mentioned in conversations?  
**Expected:** Alex (partner), Jennifer (old manager), Maria (mother), David (brother), Marcus (new manager)  
**Got:** Sam has mentioned Alex, Maria, and Marcus in conversations.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: Who are all the people Sam has mentioned in conversations?"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00024["S00024: Sam started interviewing at other…"]
    Q ==> S00024
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00028["S00028: Sam's new manager at StartupCo will be…"]
    Q ==> S00028
    S00001["S00001: Sam works as a software engineer at…"]
    S00024 -.->|link| S00001
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    S00028 -.->|link| S00027
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00028 -.->|link| S00025
```

## Q00018 — compositional_aggregation

**Q:** What different programming languages and libraries has Sam used or learned?  
**Expected:** Python, BeautifulSoup, Rust, reqwest, scraper  
**Got:** Sam has used or learned Python, Rust, and possibly others for work, but no other specific languages or libraries are mentioned besides these two.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What different programming languages and libraries has Sam…"])
    S00006["S00006: Sam started learning Spanish two months…"]
    Q ==> S00006
    S00007["S00007: Sam uses Duolingo daily for Spanish…"]
    Q ==> S00007
    S00030["S00030: Sam is now learning Rust because…"]
    Q ==> S00030
    S00017["S00017: Sam decided to use Python for the web…"]
    Q ==> S00017
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00029["S00029: Sam started at StartupCo today. The…"]
    S00030 -.->|link| S00029
    S00016["S00016: Sam started a side project: building a…"]
    S00017 -.->|link| S00016
```

## Q00019 — compositional_aggregation

**Q:** What places does Sam have travel or residence connections to?  
**Expected:** Oakland (lives), San Francisco (work), Boston (mother + planned trip), Seattle (brother), Yosemite (climbed), Joshua Tree (planned trip), Mexico (Spanish learning motivation)  
**Got:** Sam has connections to Oakland, Boston, and Seattle.  
**Correct:** ✗

```mermaid
graph TD
    Q(["Q: What places does Sam have travel or residence connections…"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00035["S00035: Alex will join Sam on the Boston trip."]
    Q ==> S00035
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    S00035 -.->|link| S00005
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00035 -.->|link| S00034
```

## Q00020 — absence_abstention

**Q:** What is Sam's favorite food?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Sam's favorite food?"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00021["S00021: Sam had an amazing climbing trip to…"]
    Q ==> S00021
    S00035["S00035: Alex will join Sam on the Boston trip."]
    Q ==> S00035
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    Q ==> S00008
    S00009["S00009: Sam has been rock climbing for three…"]
    S00021 -.->|link| S00009
    S00020["S00020: Sam's goal with the scraper is to…"]
    S00021 -.->|link| S00020
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    S00035 -.->|link| S00005
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00035 -.->|link| S00034
    S00001["S00001: Sam works as a software engineer at…"]
    S00011 -.->|link| S00001
```

## Q00021 — absence_abstention

**Q:** Does Sam have any pets?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: Does Sam have any pets?"])
    S00004["S00004: Sam lives in Oakland, California."]
    Q ==> S00004
    S00035["S00035: Alex will join Sam on the Boston trip."]
    Q ==> S00035
    S00013["S00013: Sam's brother David lives in Seattle."]
    Q ==> S00013
    S00011["S00011: Sam's mother Maria lives in Boston."]
    Q ==> S00011
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    Q ==> S00005
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00035 -.->|link| S00034
    S00001["S00001: Sam works as a software engineer at…"]
    S00011 -.->|link| S00001
```

## Q00022 — absence_abstention

**Q:** What is Sam's salary at StartupCo?  
**Expected:** unknown / not mentioned  
**Got:** unknown / not mentioned  
**Correct:** ✓

```mermaid
graph TD
    Q(["Q: What is Sam's salary at StartupCo?"])
    S00029["S00029: Sam started at StartupCo today. The…"]
    Q ==> S00029
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    Q ==> S00025
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    Q ==> S00027
    S00028["S00028: Sam's new manager at StartupCo will be…"]
    Q ==> S00028
    S00001["S00001: Sam works as a software engineer at…"]
    Q ==> S00001
    S00024["S00024: Sam started interviewing at other…"]
    S00029 -.->|link| S00024
```
