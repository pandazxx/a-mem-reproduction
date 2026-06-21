# A-Mem Memory Graph

Nodes are memories (in ingestion order). Edges are A-Mem links.
Highlighted nodes had at least one memory-evolution event.

```mermaid
graph LR
    S00001["S00001: Sam works as a software engineer at…"]
    S00002["S00002: TechCorp is a fintech company…"]
    S00003["S00003: Sam's manager at TechCorp is Jennifer."]
    S00004["S00004: Sam lives in Oakland, California."]
    S00005["S00005: Sam's partner is Alex. Alex works as a…"]
    S00006["S00006: Sam started learning Spanish two months…"]
    S00007["S00007: Sam uses Duolingo daily for Spanish…"]
    S00008["S00008: Sam goes rock climbing at Berkeley…"]
    S00009["S00009: Sam has been rock climbing for three…"]
    S00010["S00010: Sam's favorite climbing problems are…"]
    S00011["S00011: Sam's mother Maria lives in Boston."]
    S00012["S00012: Maria works as a high school chemistry…"]
    S00013["S00013: Sam's brother David lives in Seattle."]
    S00014["S00014: David works at Amazon as a product…"]
    S00015["S00015: Sam plans to visit Maria for her…"]
    S00016["S00016: Sam started a side project: building a…"]
    S00017["S00017: Sam decided to use Python for the web…"]
    S00018["S00018: Sam uses the BeautifulSoup library for…"]
    S00019["S00019: The scraper collects climbing route…"]
    S00020["S00020: Sam's goal with the scraper is to…"]
    S00021["S00021: Sam had an amazing climbing trip to…"]
    S00022["S00022: Sam climbed with Alex at Yosemite. Alex…"]
    S00023["S00023: Sam is feeling burned out at TechCorp…"]
    S00024["S00024: Sam started interviewing at other…"]
    S00025["S00025: Sam got a job offer from StartupCo, a…"]
    S00026["S00026: StartupCo is based in San Francisco but…"]
    S00027["S00027: Sam accepted the StartupCo offer and…"]
    S00028["S00028: Sam's new manager at StartupCo will be…"]
    S00029["S00029: Sam started at StartupCo today. The…"]
    S00030["S00030: Sam is now learning Rust because…"]
    S00031["S00031: Sam decided to rewrite the climbing…"]
    S00032["S00032: Sam is using the reqwest and scraper…"]
    S00033["S00033: Maria's birthday is May 15."]
    S00034["S00034: Sam booked flights to Boston for May 13…"]
    S00035["S00035: Alex will join Sam on the Boston trip."]
    S00036["S00036: David will also fly to Boston for…"]
    S00037["S00037: Sam's Spanish practice has slipped to…"]
    S00038["S00038: Sam went climbing at Berkeley Ironworks…"]
    S00039["S00039: Sam ordered a new pair of climbing…"]
    S00040["S00040: Sam is planning a long climbing trip to…"]
    S00002 --- S00001
    S00003 --- S00001
    S00005 --- S00001
    S00007 --- S00006
    S00008 --- S00004
    S00008 --- S00001
    S00009 --- S00008
    S00010 --- S00009
    S00010 --- S00008
    S00011 --- S00004
    S00011 --- S00001
    S00013 --- S00011
    S00015 --- S00011
    S00016 --- S00009
    S00016 --- S00008
    S00017 --- S00016
    S00018 --- S00017
    S00018 --- S00016
    S00018 --- S00001
    S00019 --- S00016
    S00019 --- S00017
    S00019 --- S00018
    S00020 --- S00016
    S00021 --- S00009
    S00021 --- S00020
    S00022 --- S00021
    S00023 --- S00001
    S00023 --- S00003
    S00024 --- S00001
    S00025 --- S00024
    S00025 --- S00001
    S00026 --- S00025
    S00027 --- S00025
    S00027 --- S00001
    S00028 --- S00027
    S00028 --- S00025
    S00029 --- S00025
    S00029 --- S00024
    S00029 --- S00027
    S00030 --- S00029
    S00031 --- S00020
    S00031 --- S00016
    S00032 --- S00031
    S00033 --- S00015
    S00034 --- S00015
    S00034 --- S00033
    S00035 --- S00005
    S00035 --- S00034
    S00036 --- S00034
    S00037 --- S00007
    S00038 --- S00008
    S00038 --- S00009
    S00039 --- S00009
    S00039 --- S00038
    S00040 --- S00021
    S00040 --- S00022
    S00040 --- S00009
    classDef evolved fill:#fef3c7,stroke:#d97706,stroke-width:2px
    class S00001,S00002,S00003,S00004,S00005,S00006,S00007,S00008,S00009,S00010,S00011,S00012,S00013,S00014,S00015,S00016,S00017,S00018,S00019,S00020,S00021,S00022,S00024,S00025,S00026,S00027,S00028,S00029,S00030,S00031,S00033,S00034,S00035,S00038 evolved
```
