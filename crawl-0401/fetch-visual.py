from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os
import pandas as pd
from tqdm import tqdm

def init():
    options = webdriver.ChromeOptions()

    download_path = os.path.abspath("./pdbs")
    os.makedirs(download_path, exist_ok=True)

    prefs = {
        "download.default_directory": download_path, 
        "download.prompt_for_download": False, 
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True 
    }
    options.add_experimental_option("prefs", prefs)

    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-web-security")
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--ignore-certificate-errors-spki-list")

    wd = webdriver.Chrome(options=options)
    wd.implicitly_wait(10)

    url = 'https://gpcrdb.org/structure/#'
    wd.get(url)
    original_window = wd.current_window_handle

    return wd, original_window


def clean_pages(wd, original_window):
    for handle in wd.window_handles:
        if handle != original_window:
            wd.switch_to.window(handle)
            wd.close()
    wd.switch_to.window(original_window)

def condition_check(wd, original_window, idmapping_target, i):
    flag = False
    # condition: single liangd; reference; learned target
    try:
        entrance_ligands = wd.find_elements(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[21]')
        ligand_id = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[21]').text.strip('"')
        if ligand_id in ['Apo (no ligand)', '-', 'mAb1']:
            flag = True
        if len(entrance_ligands) > 1:
            flag = True
        entrance_info = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[29]').text
        if entrance_info == '-':
            flag = True
        wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[2]/span/a').click()
        wd.switch_to.window(wd.window_handles[-1])
        target_uniprot_id = wd.find_element(By.XPATH, '/html/body/div[1]/div/div/div/main/h1/span').text
        target_uniprot_id = target_uniprot_id.split(' · ')[0]
        if target_uniprot_id not in idmapping_target:
            flag = True
        wd.close()
        wd.switch_to.window(original_window)  
        if flag == True:
            return None, None
        return target_uniprot_id, ligand_id
    except Exception as e:
            print(f"error [{str(i + 1)}] : {e}")
            return None, None

def find_pdb_code(wd):
    col_md9_elements = wd.find_elements(By.CLASS_NAME, "col-md-9")
    for element in col_md9_elements:
        parent = element.find_element(By.XPATH, "./..")
        siblings = parent.find_elements(By.XPATH, "./*") 
        if siblings[0].text.strip() == "PDB CODE":
                break
    siblings[1].find_element(By.TAG_NAME, 'a').click()
    return

def get_info(wd, original_window, idmapping_ligand_visual_dir, test_visual_dir):
    idmapping_target = pd.read_csv('idmapping_target.csv')['Target UniProt ID'].to_list()
    # pandas
    c_t = ['Target UniProt ID', 'Ligand ID', 'Label', 'Target GPCRName', 'PDB ID', 'All Info']
    v_t = []
    idmapping_ligand_dict = {}
    # Human Species select
    Species = wd.find_element(By.XPATH, '/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[1]/div/table/thead/tr[3]/th[6]/div/div/ul')
    Species.click()
    wd.find_element(By.XPATH, '/html/body/div[10]/ul/li[7]/div').click()
    # cookie remove
    wd.find_element(By.XPATH, '/html/body/div[1]/div/a').click()
    # table info
    length_table = len(wd.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr'))
    # for i in tqdm(range(912, length_table)):
    for i in tqdm(range(150, length_table)):
        # i = 149 219 427 446 808 912#TODO
        if i == 149:
            continue
        try:
        # id get
            target_gpcr_id = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[2]/span/a').text
            target_uniprot_id, ligand_id = condition_check(wd, original_window, idmapping_target, i)
            if target_uniprot_id is None:
                wd.execute_script("window.scrollBy(0, 21.68);")
                continue
            # pdb download
            pdb_id = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[8]/a').text.strip('"')
            # wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[8]/a').click()
            # wd.switch_to.window(wd.window_handles[-1])
            # find_pdb_code(wd)  # wd.find_element(By.XPATH, '/html/body/div[4]/div[8]/div[2]/a').click()  /html/body/div[4]/div[4]/div[2]/a
            # wd.switch_to.window(wd.window_handles[-1])
            # wd.find_element(By.XPATH, '/html/body/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[1]/div[2]/button').click()
            # wd.find_element(By.XPATH, '/html/body/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[1]/div[2]/ul/li[8]/a').click()
            # wd.switch_to.window(original_window)    
            # ligand info
            ligand_get = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[21]/a').click()
            ligand_get = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{str(i + 1)}]/td[21]/a').click()
            wd.switch_to.window(wd.window_handles[-1])
            ligand_get = wd.find_element(By.XPATH, '/html/body/div[4]/div[2]/div[2]/div[1]/div/div[1]/table/tbody/tr[1]/td[2]').text
            if ligand_get == 'None':
                wd.switch_to.window(original_window)
                clean_pages(wd, original_window)
                scrollable_div = wd.find_element(By.XPATH, '/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]')  
                wd.execute_script("arguments[0].scrollTop += 21.68;", scrollable_div)
                continue
            wd.switch_to.window(original_window)  
            all_info = wd.find_element(By.XPATH, f'/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]/table/tbody/tr[{i}]').text
            # restore
            idmapping_ligand_dict[ligand_id] = ligand_get
            v_t.append([target_uniprot_id, ligand_id, 1, target_gpcr_id, pdb_id, all_info])
            # clean
            clean_pages(wd, original_window)
            scrollable_div = wd.find_element(By.XPATH, '/html/body/div[4]/div[2]/div[3]/div/div/div[2]/div[2]')  
            wd.execute_script("arguments[0].scrollTop += 21.68;", scrollable_div)
        except:
            print(f'error in {i}')
            df_test = pd.DataFrame(v_t, columns=c_t)
            df_test.to_csv(test_visual_dir, index=False)
            df_mapping = pd.DataFrame(list(idmapping_ligand_dict.items()), columns=['Ligand ID', 'SMILES'])
            df_mapping.to_csv(idmapping_ligand_visual_dir, index=False)
            break
        
    df_test = pd.DataFrame(v_t, columns=c_t)
    df_test.to_csv(test_visual_dir, index=False)
    df_mapping = pd.DataFrame(list(idmapping_ligand_dict.items()), columns=['Ligand ID', 'SMILES'])
    df_mapping.to_csv(idmapping_ligand_visual_dir, index=False)
    wd.quit()

# idmapping_ligand_visual_dir = 'idmapping_ligand_visual.csv'
# test_visual_dir = 'test_visual.csv'
idmapping_ligand_visual_dir = 'idmapping_ligand_visual_x_3.csv'
test_visual_dir = 'test_visual_x_3.csv'
wd, original_window = init()
get_info(wd, original_window, idmapping_ligand_visual_dir, test_visual_dir)