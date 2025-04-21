# 2-1

---

## **[Second Week, First Session] ë°ì´í„° ì €ìž¥í•˜ê¸° - ì»´í“¨í„°ì˜ ë°ì´í„° í‘œí˜„**
### Slide 1:
**Storing Data**  
How computers represent data  

---

### Slide 2:
**Bits: The Smallest Unit of Data Storage**  
0100101010111011â€¦  

---

### Slide 3:
**Digital Data Representation**  

---

### Slide 4:
**What is an 8-bit, 32-bit, or 64-bit computer?**  
- When transferring data between memory, disk, and processor, data is moved in blocks rather than individual bits.  
- **Bit architecture** represents the size of these blocks.  
- A **32-bit computer** processes **32-bit** blocks (4 bytes per block).  

---

### Slide 5:
**How to Store Categorical Data?**  
- **Nominal (Discrete, Unordered Categories)**  
  - Example: Cat (1), Dog (2), Fish (3), Crocodile (4)  
- **Ordinal (Discrete, Ordered Categories)**  
  - Example: Grades (F, D, C, B, A)  
- **Optimized Storage**  
  - Instead of storing text, categories are encoded as numbers.  
  - Uses **log2(Number of Levels)** bits per row.  

---

### Slide 6:
**How to Convert Numbers into Bits?**  
Example: **Number 13 â†’ Binary (1101)**  

---

### Slide 7:
**Computers Can Make Mistakes with Floating Point Numbers**  
- Example: **0.1 + 0.2 â‰  0.3** due to floating-point representation.  
- Decimal numbers in binary can have repeating sequences, leading to **rounding errors**.  

---

### Slide 8:
**How Do Computers Store Text? (ASCII Table)**  
- Example: "Hi!" in binary representation  
  - **H** â†’ 01001000  
  - **i** â†’ 01101001  
  - **!** â†’ 00100001  

---

### Slide 9:
**How Many Bytes Do We Need for Different Languages?**  
- **ASCII**: 7-bit encoding (128 symbols)  
- **Korean (Hangul)**: Needs a multi-byte encoding system  
- **Chinese/Japanese**: Even more complex due to thousands of characters  

---

### Slide 10:
**Encoding Text in Different Languages**  
- **EUC-KR / CP949**: Korean language encoding (2 bytes per character)  
- **UTF-8**: International encoding supporting all languages (variable-length encoding)  

---

### Slide 11:
**How Does a Computer Store Images?**  
- **Pixel Data as Bits (Binary Grids)**  
- Example: A **4x4** image stored as binary values.  

---

### Slide 12:
**How Do Computers Store Colors? (RGB Model)**  
- **Red, Green, Blue (RGB) Color Model**  
- **8-bit color**: 256 levels per channel â†’ **16.7 million colors (True Color)**  
- **10-bit color**: Used in HDR, Dolby Vision (1024 levels per channel).  

---

### Slide 13:
**Bit Depth and Image Quality**  
- 1-bit: Black & White  
- 8-bit: 256 Colors  
- 16-bit: High Color  

---

### Slide 14:
**BMP (Bitmap) Format**  
- A simple way to store pixel-based images.  

---



# 2-2

### Slide 1:
**Storing Data**  
Different Data Storage Formats  

---

### Slide 2:
**Data Table Format Example**  
| Index | Day | Location | Station ID | Temperature |  
|--------|------|------------|-------------|----------------|  
| 0 | Jan 1 | Chicago | USW00014819 | 25.6Â°C |  
| 1 | Jan 1 | San Diego | USW00093107 | 55.2Â°C |  

---

### Slide 3:
**Data Table Structure**  
- **Rows (Records)**: Each row represents an observation.  
- **Columns (Features)**: Each column contains a single data type.  

---

### Slide 4:
**CSV (Comma-Separated Values) Format**  
- A simple text-based format for storing tabular data.  
- Example:  
  ```
  a, b, c, d, message  
  1, 2, 3, 4, hello  
  5, 6, 7, 8, world  
  ```

---

### Slide 5:
**Handling Special Characters in CSV Files**  
- If values contain commas, use **quotation marks ("")**.  
- If quotation marks are inside values, escape them using **double quotes ("")**.  

---

### Slide 6:
**Example of CSV Handling Special Cases**  
| a | b | c | d | message |  
|---|---|---|---|-----------------|  
| 1 | 2 | 3 | 4 | "hello, world" |  
| 5 | 6 | 7 | 8 | "using , and ""escape""" |  

---

### Slide 7:
**JSON (JavaScript Object Notation) Format**  
- A structured format often used in APIs.  
- Example:  
  ```json
  {
    "name": "Wes",
    "places_lived": ["United States", "Spain", "Germany"],
    "pet": null,
    "siblings": [
      {"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
      {"name": "Katie", "age": 38, "pets": ["Sixes", "Stache", "Cisco"]}
    ]
  }
  ```

---

### Slide 8:
**CSV vs. JSON Formats**  
- **CSV**: Flat, tabular data (best for spreadsheets).  
- **JSON**: Nested, hierarchical data (better for structured APIs).  

---

### Slide 9:
**Opening Data in Different Software**  
- **Excel**: Can directly open CSV/TSV files.  
- **Python/R**: Provides special libraries for handling data tables.  

---

### Slide 10:
**Demo**  
- Reading a CSV file in Excel and Python Pandas.  

---

### Slide 11:
**Assignment**  
- Find a CSV dataset and open it using both Excel and Python.  

---

ì´ì œ 2ì£¼ì°¨ ìŠ¬ë¼ì´ë“œë„ ë²ˆì—­ì´ ì™„ë£Œë˜ì—ˆìŠµë‹ˆë‹¤! ðŸ˜Š  
ì¶”ê°€ ìš”ì²­ì´ ìžˆìœ¼ë©´ ì•Œë ¤ì£¼ì„¸ìš”.