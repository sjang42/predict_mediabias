import csv


def csv2dictlist(fname, encoding='utf-8-sig'):
    """Read a csv file and convert it to list of dictionary
    :param fname: (str) file name
    :param encoding: (str) open fname with encoding
    :return: list of dictionary
    """
    with open(fname, 'r', encoding=encoding) as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        dictlist = [dict(zip(header, row)) for row in reader]

    return dictlist


def dictlist2csv(dict_list, out_name='sample.csv', encoding='utf-8-sig'):
    header = list(dict_list[0].keys())

    with open(out_name, 'w', encoding=encoding) as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for dic in dict_list:
            wr.writerow(list(dic.values()))
