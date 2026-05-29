# Astro Stacker 2.3 - uživatelský manuál

Astro Stacker slouží ke skládání astronomických snímků z DSLR, bezzrcadlovek, astronomických kamer a chytrých dalekohledů. Umí načíst sekvenci Light snímků, volitelně použít Flat/Bias/Dark kalibraci, zarovnat snímky podle hvězd nebo komety, složit výsledek a nabídnout základní úpravy náhledu pro kontrolu a export.

Program pracuje tak, aby lineární FIT výstup zůstal vhodný pro další zpracování. Většina úprav v pravém panelu ovlivňuje hlavně náhled a export do obrazových formátů, ne původní lineární data.

## 1. Co program umí

- Skládání Light snímků ze složky.
- Podpora FIT/FITS, běžných RAW formátů fotoaparátů a obrazových formátů.
- Volitelná kalibrace pomocí Flat, Bias a Dark snímků.
- Zarovnání posunem, ECC afinně, podle hvězd přes RANSAC, perspektivně podle hvězd nebo podle komety.
- Automatický nebo ruční výběr referenčního snímku.
- Filtrování nejlepších snímků podle skóre kvality.
- Diagnostická tabulka kvality snímků.
- Uložení a načtení profilů nastavení.
- Náhled s histogramem, křivkami, barvami, vyvážením pozadí a funkcí Balance.
- Export lineárního FIT nebo vizuálně upraveného PNG/TIFF.
- Spuštění wrapperu pro PixInsight.

## 2. Rychlý postup pro běžné skládání

1. Klikni na **Choose folder / Vybrat složku** a vyber složku s Light snímky.
2. Pokud jsou ve stejné složce i JPG/PNG/BMP/TIFF náhledy, zapni **RAW only / Pouze RAW**.
3. Pro běžnou deep-sky sekvenci nech nastaveno:
   - **Alignment / Zarovnání:** Star alignment - stars + RANSAC
   - **Stacking / Skládání:** Sigma-clipped mean
   - **Auto reference / Automaticky vybrat nejlepší referenci:** zapnuto
   - **Use only best frames / Použít jen nejlepší snímky:** podle potřeby zapnuto
4. Klikni na **Start stacking / Spustit skládání**.
5. Po dokončení zkontroluj náhled, diagnostickou tabulku a hlášení o počtu použitých/vyřazených snímků.
6. Výsledek ulož přes **File > Save result as / Soubor > Uložit výsledek jako**.

## 3. Levý panel - skládání

### Výběr složky a náhledu

**Choose folder / Vybrat složku** načte pracovní složku se snímky. Program v seznamu ukazuje Light, Flat, Bias a Dark snímky. Po stackování se v seznamu označí:

- `*` referenční snímek,
- `x` vyřazený snímek.

Rozbalovací seznam slouží také k rychlé kontrole jednotlivých snímků.

### RAW only / Pouze RAW

Tato volba při stackování ponechá FIT/FITS a foto RAW soubory, například ARW, CR2, CR3 apod. Vynechá JPG, PNG, BMP a TIFF. Hodí se, když chytrý dalekohled nebo fotoaparát ukládá do složky současně pracovní snímky i náhledy.

### Zarovnání

**Translation / Pouze posun**  
Rychlé zarovnání pouze posunem. Vhodné pro malé drifty bez rotace.

**Calibration/no alignment / Kalibrační snímky - bez zarovnání**  
Používá se pro skládání kalibračních snímků pixel na pixel.

**ECC affine / Afinní ECC**  
Zarovnání posunem, rotací a změnou měřítka pomocí obrazové korelace. Může být pomalejší a u slabých objektů nemusí být ideální.

**Star alignment + RANSAC**  
Hlavní doporučený režim. Program detekuje hvězdy, páruje je s referenčním snímkem a přes RANSAC odhadne transformaci. Snímky bez platného zarovnání vyřadí.

**Comet alignment**  
Skládá na kometu podle označené polohy komety.

**Star + Comet**  
Vytvoří samostatný výstup pro hvězdy a kometu.

### Automatická a ruční reference

**Automatically choose best reference / Automaticky vybrat nejlepší referenci** vybere snímek s nejlepším skóre kvality. Skóre vychází hlavně z ostrosti a počtu detekovaných hvězd.

Pokud automatická reference u některé sekvence nefunguje ideálně, vyber v seznamu vhodný snímek a klikni na **Use current frame as reference / Použít aktuální snímek jako referenci**. Tím se automatická reference vypne a program použije zvolený snímek.

### Quality filter / Použít jen nejlepší snímky

Program spočítá skóre kvality pro každý Light snímek. Skóre kombinuje:

- ostrost snímku,
- počet detekovaných hvězd.

Volba **Keep / Ponechat** určuje, kolik procent nejlepších snímků se použije. Například 80 % znamená, že nejhorších 20 % bude vyřazeno ještě před alignmentem.

Důležité: snímek může projít filtrem kvality, ale později být vyřazen, pokud selže zarovnání.

### Max. star drift / Max. drift hvězd

Určuje největší očekávaný posun hvězd vůči referenci. U běžných sekvencí stačí menší hodnota. U EAA, ditheringu nebo chytrých dalekohledů může být potřeba hodnotu zvýšit.

### Ignore border / Ignorovat okraj

Program při hledání hvězd ignoruje okraje snímku. To pomáhá, pokud jsou na okraji větve, šum, vinětace, černé rohy po rotaci nebo jiné rušivé struktury.

### Strict star filter / Přísný filtr hvězd

Přísnější kontrola tvaru hvězd. Pomáhá proti větvím, stopám a protáhlým artefaktům. Pokud program odmítá příliš mnoho použitelných snímků, může někdy pomoci tuto volbu vypnout.

## 4. Diagnostická tabulka snímků

Ve spodní části je tabulka **Frame quality / Vyhodnocení snímků**. Slouží ke kontrole, proč byly snímky použity nebo vyřazeny.

Sloupce:

- `#` pořadí snímku,
- `File / Soubor` název souboru,
- `Score / Skóre` celkové skóre kvality,
- `Stars / Hvězdy` počet detekovaných hvězd,
- `Sharpness / Ostrost` ostrost podle Laplacian variance,
- `Status / Stav` použití snímku,
- `Ref` označení reference.

Typické stavy:

- **Used / Použit** - snímek byl složen.
- **Reference** - referenční snímek.
- **Excluded by quality / Vyřazen kvalitou** - snímek neprošel filtrem kvality.
- **Rejected alignment / Selhal alignment** - snímek byl kvalitní dost, ale nepodařilo se ho zarovnat.
- **Skipped / Vynechán** - snímek nebyl součástí stacku, například kalibrační nebo nepoužitý typ.

Kliknutí na řádek tabulky přepne náhled na daný snímek.

### Kontrola snímků před skládáním

Volba **Review frames before stacking / Zkontrolovat snímky před skládáním** vloží před samotné skládání ruční kontrolní krok.

Postup:

1. Zapni **Zkontrolovat snímky před skládáním**.
2. Klikni na **Spustit skládání**.
3. Program pouze spočítá kvalitu snímků, vybere referenci a naplní tabulku Frame Quality.
4. V tabulce vyber snímek, který nechceš použít.
5. Stiskni mezerník. Stav se změní na **Vyřazený** a řádek zčervená.
6. Dalším stiskem mezerníku lze snímek znovu povolit.
7. Klikni na **Pokračovat ve skládání**.

Referenční snímek nelze mezerníkem vyřadit. Pokud chceš vyřadit referenci, nejprve vyber jiný referenční snímek.

Ruční vyřazení má přednost před automatickým výběrem. Automatický quality filter stále určí první výběr, ale před finálním alignmentem můžeš ručně odebrat další snímky.

## 5. Kalibrace Flat, Bias a Dark

Program umí použít kalibrační snímky dvěma způsoby:

- automaticky, pokud jsou ve vhodně pojmenovaných složkách,
- ručně přes tlačítka **Flat**, **Bias** a **Dark** v pravém panelu.

Kalibrace se aplikuje před zarovnáním a stackováním Light snímků.

Tlačítko **Reset calibration / Reset kalibrace** odstraní vybrané kalibrační snímky.

## 6. Skládání komety

Pro kometu je nejspolehlivější označit její polohu ve dvou snímcích.

1. Vyber složku se sekvencí.
2. Klikni na **Comet First / Kometa první**.
3. Program načte první snímek. Klikni v náhledu na jádro komety.
4. Klikni na **Comet Last / Kometa poslední**.
5. Program načte poslední snímek. Klikni na jádro komety.
6. Program zná pohyb komety mezi začátkem a koncem sekvence.
7. Vyber režim **Comet alignment** nebo **Star + Comet**.
8. Spusť skládání.

**Comet Clear / Kometa smaž** smaže uloženou první i poslední polohu komety.

Volby pro kometu:

- **Max. comet motion / Max. pohyb komety** - největší očekávaný pohyb komety.
- **Refine comet position / Jemně doladit kometu** - dohledá jádro lokální korelací.
- **Comet template / Šablona komety** - velikost šablony kolem jádra.
- **Comet search / Hledání komety** - jak daleko od předpokládané polohy se smí jádro dohledat.

## 7. Předvolby a profily nastavení

Předvolby se ukládají jako JSON profil.

Použij:

- **File > Save settings profile / Soubor > Uložit profil nastavení**
- **File > Load settings profile / Soubor > Načíst profil nastavení**

Profil ukládá nastavení stackování, kalibrace, komety i pravého panelu. Hodí se například pro různé typy dat:

- DSLR RAW,
- Seestar/Vespera EAA,
- komety,
- kalibrační snímky,
- čistý lineární FIT workflow.

## 8. Pravý panel - náhled a úpravy

Pravý panel slouží ke kontrole a vizuálním úpravám výsledku. Tyto úpravy jsou určené hlavně pro náhled a export do PNG/TIFF. Lineární FIT výstup zůstává vhodný pro další zpracování.

### Show stacked image / Zobraz složený obraz

Vrátí náhled na původní složený obraz bez ořezu, neutralizace a vizuálních úprav.

### Balance

Tlačítko **Balance** automaticky nastaví rozumný náhled lineárního snímku:

- neutralizuje pozadí pro náhled,
- nastaví black point,
- nastaví gamma,
- snaží se zobrazit slabý lineární obraz čitelněji.

Balance nemění lineární FIT data. Je to rychlá pomůcka pro náhled.

### Black point, White point, Gamma

Základní křivky pro zobrazení:

- **Black point** posouvá černý bod.
- **White point** určuje světlý bod.
- **Gamma** mění střední tóny.

### Highlight compression / Komprese jasů

Potlačuje přepaly jasných hvězd a jader objektů. Hodí se pro náhled snímků s velmi jasnými hvězdami.

### Vignette removal / Odstranění vinětace

Jemně zesvětlí okraje a rohy. Není to náhrada skutečného Flat snímku, ale může pomoci pro rychlý vizuální náhled.

### Synthetic flat / Umělý flat

Odhadne hladké pozadí ze složeného obrazu a použije ho jako jemnou korekci. Používej opatrně, zvlášť u rozsáhlých mlhovin, kde může program část mlhoviny považovat za pozadí.

### Color background correction / Korekce barevného pozadí

Potlačuje hladký barevný závoj pozadí po RGB kanálech. Je užitečná hlavně u velmi barevně nevyvážených snímků, například růžové nebo fialové pozadí z chytrých dalekohledů.

### SCNR Green

Potlačuje zelený nádech. Používej jen tehdy, pokud je zelená opravdu rušivá.

### Contrast, Saturation, RGB

- **Contrast / Kontrast** mění kontrast náhledu.
- **Saturation / Saturace** mění barevnost.
- **Red, Green, Blue / Červená, Zelená, Modrá** ručně upravují jednotlivé kanály.

### Histogram

Histogram ukazuje jasový kanál a RGB kanály. Pomáhá kontrolovat, zda nejsou data příliš oříznutá v černé nebo bílé.

### Crop edges / Oříznout okraje

Ořízne zadané procento z každého okraje aktuálního snímku. Hodí se po rotaci, ditheringu nebo EAA sekvencích s tmavými rohy.

### Auto White Balance

Pokusí se automaticky vyvážit bílé/šedé oblasti. U některých astrofotografií může být vhodnější ruční úprava RGB nebo korekce barevného pozadí.

### Neutralize background / Neutralizovat pozadí

Snaží se srovnat barvu pozadí. Nejlépe funguje, když je v obraze dost neutrálního pozadí bez mlhovin a bez špatných okrajů.

### Flip horizontal / vertical

Překlopí náhled horizontálně nebo vertikálně.

### Fit a 1:1

- **Fit** přizpůsobí snímek oknu.
- **1:1** zobrazí skutečné pixely.

## 9. PixInsight wrapper

Tlačítko **PixInsight** otevře PixInsight a spustí AS_Stacker wrapper, pokud je správně dostupný. Wrapper umožňuje spustit CLI verzi stackeru přímo z PixInsightu a po dokončení otevřít výsledný FIT v PixInsightu.

Pro wrapper je potřeba nastavit Python prostředí, ve kterém jsou nainstalované potřebné balíčky.

## 10. Doporučené postupy

### Běžná deep-sky sekvence

- Star alignment + RANSAC
- Sigma-clipped mean
- Auto reference zapnuto
- Quality filter 80-100 %
- Strict star filter zapnuto

### Sekvence s velkým driftem nebo ditheringem

- Zvyš Max. star drift.
- Zkus ruční referenci ze středu sekvence.
- Sleduj diagnostickou tabulku, jestli snímky padají na kvalitě nebo alignmentu.

### Chytré dalekohledy a EAA

- Zapni RAW only, pokud jsou ve složce i náhledové obrázky.
- U rotace a tmavých rohů použij ořez.
- U růžového/fialového pozadí použij Color background correction.

### Komety

- Označ první i poslední polohu komety.
- Použij Comet alignment nebo Star + Comet.
- U slabého jádra nech zapnuté jemné doladění komety.

## 11. Co dělat při problémech

### Program vyřazuje mnoho snímků

Zkontroluj tabulku Frame quality:

- pokud jsou snímky **Excluded by quality**, sniž filtr kvality nebo zvyš Keep %;
- pokud jsou **Rejected alignment**, zvyš Max. star drift, změň referenci nebo zkus jiný režim zarovnání.

### Snímek je barevně divný

Zkus postupně:

1. Balance,
2. Color background correction,
3. Neutralize background,
4. ruční RGB posuny,
5. případně ořez okrajů a neutralizaci znovu.

### Výsledek má tmavé rohy

Je to běžné u rotace, ditheringu nebo velkého driftu. Použij Crop edges nebo výsledek dále ořízni v externím editoru.

### FIT výstup vypadá tmavě

To je normální. FIT je lineární a potřebuje stretch. Použij Balance pro náhled nebo výstup zpracuj v PixInsightu, Sirilu či jiném astrofoto editoru.
