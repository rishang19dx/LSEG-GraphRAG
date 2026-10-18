"""Pre-packaged semiconductor and hardware supply chain corpus builder."""

import json
from pathlib import Path
from typing import List, Dict, Any
from lseg_graphrag.schema.provenance import DocumentMetadata, ChunkMetadata
from lseg_graphrag.ingestion.chunker import SectionAwareChunker


def get_semiconductor_seed_documents() -> List[Dict[str, Any]]:
    """Returns curated 10-K disclosures and news for the semiconductor universe."""
    return [
        {
            "metadata": {
                "doc_id": "SEC_10K_NVDA_2024",
                "company_symbol": "NVDA",
                "company_name": "NVIDIA Corporation",
                "cik": "0001045810",
                "form_type": "10-K",
                "filing_date": "2024-02-21",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
            },
            "sections": {
                "item_1": (
                    "NVIDIA Corporation pioneered GPU-accelerated computing to help solve challenging computational problems. "
                    "We specialize in products and platforms for Compute & Networking and Graphics. "
                    "We utilize a fabless manufacturing model, partnering with independent foundries for the production of our semiconductor wafers. "
                    "Taiwan Semiconductor Manufacturing Company (TSMC) manufactures our advanced semiconductor wafers, including our Hopper, Ada Lovelace, and Blackwell architectures, utilizing cutting-edge 4N and 3nm process technologies. "
                    "We also utilize third-party packaging partners, including TSMC's CoWoS (Chip-on-Wafer-on-Substrate) advanced packaging services, as well as Amkor Technology and Siliconware Precision Industries (SPIL). "
                    "Our GPU architectures integrate advanced high-bandwidth memory (HBM3 and HBM3e) supplied by Micron Technology, SK Hynix, and Samsung Electronics. "
                    "Our major customers include hyperscale cloud providers, server original equipment manufacturers (OEMs), and enterprises such as Microsoft Corporation, Amazon Web Services, Alphabet Inc., and Meta Platforms. "
                    "We compete with Advanced Micro Devices (AMD) in data center GPUs and gaming graphics, and with Intel Corporation in server processors and accelerator architectures. "
                    "We also compete with custom ASIC silicon developed internally by cloud service providers."
                ),
                "item_1a": (
                    "We depend heavily on third-party foundries, particularly TSMC in Taiwan, to manufacture our semiconductor wafers. "
                    "A disruption in foundry manufacturing, whether due to capacity constraints, equipment breakdowns, power outages, seismic events, or geopolitical tensions in Taiwan, would materially harm our business and financial results. "
                    "Substantially all of our advanced computing GPUs are packaged by TSMC using specialized CoWoS advanced packaging. Capacity shortages in advanced packaging can limit our ability to meet surging market demand. "
                    "We are exposed to risks from international trade policies and export control regulations administered by the U.S. Department of Commerce Bureau of Industry and Security (BIS). "
                    "In October 2022 and October 2023, the U.S. government implemented updated export controls restricting the shipment of advanced AI processors (including our A100, H100, A800, and H800 products) and semiconductor manufacturing equipment to China and certain other designated destinations without a license. "
                    "Furthermore, our supply chain is exposed to raw material constraints, including rare gases like neon, semiconductor grade silicon, and substrate materials."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF NVIDIA CORPORATION\n"
                    "1. Mellanox Technologies, Ltd. - Jurisdiction: Israel\n"
                    "2. NVIDIA International Holdings B.V. - Jurisdiction: Netherlands\n"
                    "3. NVIDIA Singapore PTE. LTD. - Jurisdiction: Singapore\n"
                    "4. NVIDIA Development UK Ltd - Jurisdiction: United Kingdom\n"
                    "5. NVIDIA Semiconductor Technology (Shanghai) Co., Ltd. - Jurisdiction: China\n"
                    "6. NVIDIA Graphics Private Limited - Jurisdiction: India\n"
                    "7. Cumulus Networks, Inc. - Jurisdiction: Delaware, United States"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "SEC_10K_TSM_2024",
                "company_symbol": "TSM",
                "company_name": "Taiwan Semiconductor Manufacturing Company Limited",
                "cik": "0001046179",
                "form_type": "20-F",
                "filing_date": "2024-04-10",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/1046179/000104617924000007/tsm-20231231.htm",
            },
            "sections": {
                "item_1": (
                    "Taiwan Semiconductor Manufacturing Company Limited (TSMC) is the world's largest dedicated pure-play semiconductor foundry. "
                    "We manufacture semiconductor chips for our customers based on their proprietary integrated circuit designs. "
                    "Our customers include global technology leaders such as Apple Inc., NVIDIA Corporation, Advanced Micro Devices (AMD), Qualcomm Incorporated, Broadcom Inc., and MediaTek Inc. "
                    "Apple Inc. and NVIDIA Corporation represent our largest commercial customers by wafer revenue. "
                    "To support advanced manufacturing down to 3nm and 2nm nodes, we rely on photolithography equipment suppliers. "
                    "ASML Holding N.V. is our primary and sole supplier of Extreme Ultraviolet (EUV) lithography scanners, which are essential for manufacturing leading-edge nodes. "
                    "We also procure etch, deposition, and wafer cleaning systems from Applied Materials, Inc., Lam Research Corporation, Tokyo Electron Limited, and KLA Corporation."
                ),
                "item_1a": (
                    "The majority of our primary wafer fabrication facilities (GigaFabs) are located in Taiwan, specifically in the Hsinchu Science Park, Central Taiwan Science Park in Taichung, and Southern Taiwan Science Park in Tainan. "
                    "Taiwan is susceptible to natural disasters including earthquakes, typhoons, and periodic regional water and electrical power shortages. "
                    "In April 2024, a major earthquake struck off the coast of Hualien, Taiwan, temporarily impacting operations and causing tool recalibrations. "
                    "Furthermore, cross-strait geopolitical tensions between Taiwan and the People's Republic of China represent a continuing operational risk. "
                    "To diversify geographic exposure, we are expanding global manufacturing operations through TSMC Arizona in the United States, JASM in Kumamoto, Japan, and ESMC in Dresden, Germany. "
                    "We are also exposed to export restrictions imposed by international governments that limit our ability to fabricate certain advanced microprocessors for restricted entities in China."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY\n"
                    "1. TSMC North America - Jurisdiction: California, United States\n"
                    "2. TSMC Arizona Corporation - Jurisdiction: Arizona, United States\n"
                    "3. Japan Advanced Semiconductor Manufacturing, Inc. (JASM) - Jurisdiction: Kumamoto, Japan\n"
                    "4. TSMC Europe B.V. - Jurisdiction: Amsterdam, Netherlands\n"
                    "5. TSMC China Company Limited - Jurisdiction: Shanghai, China\n"
                    "6. TSMC Nanjing Company Limited - Jurisdiction: Nanjing, China\n"
                    "7. European Semiconductor Manufacturing Company (ESMC) GmbH - Jurisdiction: Dresden, Germany"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "SEC_10K_AAPL_2024",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "cik": "0000320193",
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019324000106/aapl-20240928.htm",
            },
            "sections": {
                "item_1": (
                    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, and sells a variety of related services. "
                    "Our key products include the iPhone, Mac, iPad, Apple Watch, and AirPods. "
                    "We utilize custom-designed silicon architectures, including A-series mobile processors and M-series computing chips. "
                    "We outsource wafer fabrication of all our proprietary Apple Silicon chips exclusively to Taiwan Semiconductor Manufacturing Company (TSMC). "
                    "We procure display panels from Samsung Display and LG Display, and memory components from Micron Technology, Samsung Electronics, and Kioxia Corporation. "
                    "Substantially all of our final product assembly is performed by third-party manufacturing partners in Asia. "
                    "Hon Hai Precision Industry Co., Ltd. (Foxconn) is our primary assembly partner for iPhone products, operating major manufacturing campuses in Zhengzhou and Shenzhen, China, as well as facilities in Tamil Nadu, India. "
                    "Pegatron Corporation and Luxshare Precision Industry also provide assembly manufacturing services."
                ),
                "item_1a": (
                    "A significant portion of our manufacturing, assembly operations, and supply chain is concentrated in the Asia-Pacific region, notably in China, Taiwan, India, and Vietnam. "
                    "Disruptions in these regions resulting from labor disputes, factory shutdowns, geopolitical tensions, trade disputes, or natural hazards such as earthquakes in Taiwan could delay product launches and reduce revenues. "
                    "We are exposed to trade policies and tariff measures between the United States and China. "
                    "Because we single-source our proprietary A-series and M-series chips from TSMC, any disruption to TSMC's manufacturing facilities would directly impair our ability to supply iPhones and Macs."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF APPLE INC.\n"
                    "1. Apple Operations International Ltd - Jurisdiction: Ireland\n"
                    "2. Apple Operations Europe Ltd - Jurisdiction: Ireland\n"
                    "3. Apple Distribution International Ltd - Jurisdiction: Ireland\n"
                    "4. Apple Japan Inc. - Jurisdiction: Tokyo, Japan\n"
                    "5. Apple Computer Trading (Shanghai) Co., Ltd. - Jurisdiction: Shanghai, China\n"
                    "6. Apple (UK) Limited - Jurisdiction: United Kingdom\n"
                    "7. Bedrock Manufacturing LLC - Jurisdiction: Delaware, United States"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "SEC_10K_ASML_2024",
                "company_symbol": "ASML",
                "company_name": "ASML Holding N.V.",
                "cik": "0000937966",
                "form_type": "20-F",
                "filing_date": "2024-02-28",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/937966/000093796624000012/asml-20231231.htm",
            },
            "sections": {
                "item_1": (
                    "ASML Holding N.V. is the world leader in photolithography systems for the semiconductor industry. "
                    "We manufacture Deep Ultraviolet (DUV) and Extreme Ultraviolet (EUV) lithography equipment, including High-NA EUV platforms. "
                    "Our lithography systems enable semiconductor manufacturers to image intricate circuit patterns on silicon wafers. "
                    "Our largest customers are leading semiconductor foundries and memory manufacturers, including TSMC, Intel Corporation, Samsung Electronics, and Micron Technology. "
                    "We depend on critical single-source suppliers for sub-assemblies. "
                    "Carl Zeiss SMT GmbH is our exclusive supplier of high-precision optical lenses, mirrors, and collector systems required for our EUV and DUV scanners. "
                    "We operate main manufacturing and R&D facilities in Veldhoven, Netherlands, alongside facilities in San Diego, California, and Wilton, Connecticut."
                ),
                "item_1a": (
                    "We operate in a highly concentrated supply chain and customer landscape. "
                    "A disruption at Carl Zeiss in Germany would immediately halt our ability to ship EUV lithography machines. "
                    "We are subject to strict export control regulations by the Netherlands government and the United States government. "
                    "Under updated Dutch export licensing rules and U.S. extraterritorial export regulations, ASML is prohibited from shipping our most advanced immersion DUV lithography systems (such as the Twinscan NXT:2000i series) and all EUV systems to customers in China without prior government authorization. "
                    "Geopolitical trade conflicts could reduce our revenue from Chinese semiconductor customers."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF ASML HOLDING N.V.\n"
                    "1. ASML Netherlands B.V. - Jurisdiction: Netherlands\n"
                    "2. ASML US, LLC - Jurisdiction: Delaware, United States\n"
                    "3. Cymer LLC - Jurisdiction: California, United States\n"
                    "4. ASML Hong Kong Limited - Jurisdiction: Hong Kong\n"
                    "5. ASML Korea Co. Ltd. - Jurisdiction: Hwaseong, South Korea\n"
                    "6. ASML Taiwan Joint Services Co., Ltd. - Jurisdiction: Hsinchu, Taiwan\n"
                    "7. Berliner Glas KGaA - Jurisdiction: Berlin, Germany"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "SEC_10K_AMD_2024",
                "company_symbol": "AMD",
                "company_name": "Advanced Micro Devices, Inc.",
                "cik": "0000002488",
                "form_type": "10-K",
                "filing_date": "2024-02-05",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/2488/000000248824000012/amd-20231230.htm",
            },
            "sections": {
                "item_1": (
                    "Advanced Micro Devices, Inc. (AMD) operates as a global semiconductor company. "
                    "Our segments include Data Center, Client, Gaming, and Embedded. "
                    "We offer x86 microprocessors (EPYC server CPUs and Ryzen desktop/notebook CPUs), Instinct AI data center GPU accelerators, and Adaptive SoCs (Xilinx platforms). "
                    "We operate as a fabless company and outsource all wafer fabrication. "
                    "TSMC is our principal foundry partner, manufacturing our advanced EPYC processors and Instinct MI300 series accelerators using 5nm, 4nm, and 3nm nodes. "
                    "We also partner with GLOBALFOUNDRIES for certain legacy trailing node products. "
                    "Our Instinct MI300 accelerators integrate HBM3 high-bandwidth memory sourced from Micron Technology and SK Hynix. "
                    "We compete directly with Intel Corporation in CPUs and with NVIDIA Corporation in GPUs and AI accelerators."
                ),
                "item_1a": (
                    "We rely heavily on TSMC as our sole foundry for our leading-edge microprocessors and graphics processors. "
                    "If TSMC experiences operational disruptions, capacity shortages, or supply interruptions in Taiwan due to earthquakes or geopolitical conflicts, our shipments would be severely impacted. "
                    "U.S. government export controls have restricted sales of high-performance accelerators, including the Instinct MI300X, to customers in China. "
                    "We also face risks regarding customer concentration in the hyperscale cloud and PC OEM sectors."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF ADVANCED MICRO DEVICES, INC.\n"
                    "1. Xilinx, Inc. - Jurisdiction: Delaware, United States\n"
                    "2. AMD International Sales & Service Ltd - Jurisdiction: Delaware, United States\n"
                    "3. Advanced Micro Devices (Singapore) Pte Ltd - Jurisdiction: Singapore\n"
                    "4. AMD Technology (China) Co., Ltd. - Jurisdiction: Beijing, China\n"
                    "5. AMD India Private Limited - Jurisdiction: Bengaluru, India\n"
                    "6. Pensando Systems Inc. - Jurisdiction: Delaware, United States"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "SEC_10K_INTC_2024",
                "company_symbol": "INTC",
                "company_name": "Intel Corporation",
                "cik": "0000050863",
                "form_type": "10-K",
                "filing_date": "2024-01-26",
                "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/50863/000005086324000010/intc-20231230.htm",
            },
            "sections": {
                "item_1": (
                    "Intel Corporation is a global leader in designing and manufacturing semiconductors. "
                    "Unlike fabless peers, we operate an Integrated Device Manufacturer (IDM) model through Intel Foundry. "
                    "We operate primary wafer manufacturing fabs in Oregon, Arizona, New Mexico, and internationally in Leixlip, Ireland and Kiryat Gat, Israel. "
                    "We are also building new fabs in Magdeburg, Germany, and Columbus, Ohio. "
                    "For our manufacturing operations, we purchase advanced lithography scanners from ASML Holding N.V., including being the first customer to receive ASML's High-NA EUV (Twinscan EXE:5000) system. "
                    "We also selectively outsource wafer production for specific tiles (such as Arrow Lake and Lunar Lake compute tiles) to TSMC. "
                    "We compete with AMD in PC and server processors, and with NVIDIA in data center accelerators."
                ),
                "item_1a": (
                    "Our manufacturing facilities require substantial capital expenditure. "
                    "Our operations in Israel (Fab 28 in Kiryat Gat) are exposed to regional security risks and conflict in the Middle East. "
                    "We are dependent on key equipment suppliers including ASML, Applied Materials, and Lam Research. "
                    "U.S. export regulations restricting the sale of advanced compute chips and semiconductor manufacturing equipment to China impact our revenue potential in the Chinese market."
                ),
                "exhibit_21": (
                    "EXHIBIT 21: SUBSIDIARIES OF INTEL CORPORATION\n"
                    "1. Mobileye Global Inc. - Jurisdiction: Delaware, United States\n"
                    "2. Intel Ireland Limited - Jurisdiction: Cayman Islands\n"
                    "3. Intel Semiconductor (Dalian) Ltd. - Jurisdiction: China\n"
                    "4. Intel Electronics Ltd. - Jurisdiction: Israel\n"
                    "5. Altera Corporation - Jurisdiction: Delaware, United States\n"
                    "6. Intel Technology Asia Pte Ltd - Jurisdiction: Singapore"
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "NEWS_2024_04_TAIWAN_QUAKE",
                "company_symbol": "TSM",
                "company_name": "Reuters Supply Chain Intelligence",
                "cik": None,
                "form_type": "News",
                "filing_date": "2024-04-04",
                "source_url": "https://www.reuters.com/technology/taiwan-quake-chip-supply-impact-2024-04-04/",
            },
            "sections": {
                "news_article": (
                    "TAIPEI/SAN FRANCISCO - A powerful 7.4-magnitude earthquake struck Taiwan's eastern coast near Hualien on Wednesday. "
                    "Taiwan Semiconductor Manufacturing Company (TSMC), the premier chipmaker for Apple and NVIDIA, evacuated fabrication personnel across its Hsinchu, Taichung, and Tainan science parks. "
                    "TSMC reported that its critical Extreme Ultraviolet (EUV) lithography tools, supplied by Dutch firm ASML, suffered no structural damage, though automated safety systems triggered emergency shutoffs and tool recalibrations. "
                    "The event underscores the acute geographic exposure of the global technology sector to Taiwan. "
                    "Both Apple, which relies exclusively on TSMC for iPhone processors, and NVIDIA, which depends on TSMC's CoWoS packaging in Taiwan for Hopper AI GPUs, confirmed they were closely monitoring potential supply bottlenecks."
                ),
            },
        },
        {
            "metadata": {
                "doc_id": "NEWS_2024_06_DUTCH_EXPORT_RULES",
                "company_symbol": "ASML",
                "company_name": "Financial Times Tech Dispatch",
                "cik": None,
                "form_type": "News",
                "filing_date": "2024-06-18",
                "source_url": "https://www.ft.com/content/asml-export-restrictions-update-2024",
            },
            "sections": {
                "news_article": (
                    "THE HAGUE/AMSTERDAM - The Dutch government announced updated export licensing guidelines aligned with U.S. national security measures. "
                    "Under the expanded restrictions, ASML Holding N.V. is prohibited from exporting advanced immersion DUV lithography systems to Chinese foundries without explicit ministerial permits. "
                    "ASML's EUV lithography machines have been barred from shipment to China since 2019. "
                    "The regulatory actions affect Chinese semiconductor manufacturers seeking to produce advanced logic chips below 7-nanometer thresholds. "
                    "Key equipment peers Applied Materials and Lam Research also indicated compliance with parallel U.S. Bureau of Industry and Security export rules."
                ),
            },
        },
    ]


def build_and_save_sample_corpus(
    output_dir: str = "data/processed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[ChunkMetadata]:
    """Generates and writes parsed chunks and document metadata for the seed corpus."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunker = SectionAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[ChunkMetadata] = []
    docs_catalog: List[Dict[str, Any]] = []

    seed_docs = get_semiconductor_seed_documents()

    for item in seed_docs:
        doc_meta = DocumentMetadata(**item["metadata"])
        docs_catalog.append(doc_meta.model_dump())

        for section_key, text in item["sections"].items():
            section_title = {
                "item_1": "Item 1: Business",
                "item_1a": "Item 1A: Risk Factors",
                "exhibit_21": "Exhibit 21: Subsidiaries",
                "news_article": "News Article",
            }.get(section_key, section_key.replace("_", " ").title())

            section_chunks = chunker.chunk_section(
                doc_meta=doc_meta,
                section_key=section_key,
                section_title=section_title,
                text=text,
            )
            all_chunks.extend(section_chunks)

    # Save documents catalog
    docs_file = out_path / "documents.json"
    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(docs_catalog, f, indent=2)

    # Save chunks index
    chunks_file = out_path / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, indent=2)

    return all_chunks
