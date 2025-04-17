import os



cwd = os.getcwd()

save_name = 'cusfontlist.txt'
fonts_dir = './englishfonts'




def recursion_dir_all_file(path):
    file_list = []
    for dir_path, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(dir_path, file)
            if "\\" in file_path:
                file_path = file_path.replace('\\', '/')
            file_list.append(file_path)
        for dir in dirs:
            file_list.extend(recursion_dir_all_file(os.path.join(dir_path, dir)))
    return file_list

if __name__ == '__main__':
    file_lst = recursion_dir_all_file(fonts_dir)
    file_lst = [el for el in file_lst if el.endswith('otf') or el.endswith('ttf')]
    # print(file_lst)
    file = '\n'.join(file_lst)
    with open(save_name,'w') as fw:
        fw.write(file)
