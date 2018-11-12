
def expand_attr(attr, class_name, dataset_path):
    """
    """
    # 得到类别名称和对应的属性后再进行扩充
    class_img_count = img_count(dataset_path, class_name)
    # 得到图片的数量然后进行扩充
    expanded_attr = np.tile(attr, (class_img_count, 1))

    return expanded_attr

def read_all_attrs(dataset_path):
    """Get all the attribute vectors.

    Args:
        dataset_path: the path where the data stored

    Returns:
        all_attrs: all the attributes, one class one vector
    """
    attr_file_path = dataset_path + '/' + r'predicate-matrix-binary.txt'
    attr_file = io.open(attr_file_path, 'r')

    all_attrs = []

    attr_count = 0
    while attr_count != 50:
        attr = np.array(attr_file.readline().split(' '), dtype=int)
        all_attrs.append(attr)
        attr_count += 1

    attr_file.close()

    return all_attrs

def read_and_expand_split_attr(dataset_path, split_name):
    """Get attributes of a split of classes.

    Args:
        dataset_path: 
        split_name:

    Returns:
        split_attrs: 
    """
    all_img_path = dataset_path + r'/JPEGImages'

    all_attrs = read_all_attrs(dataset_path)

    all_class_name = get_class_name(dataset_path, 'all')
    split_class_name = get_class_name(dataset_path, split_name)

    # 下面功能的实现的前提是：classes.txt中的类别和predicate-matrix-binary.txt中的属性向量是一一对应的
    attr_count = 1
    for class_name in split_class_name:
        attr_index = all_class_name.index(class_name)
        correspond_attr = all_attrs[attr_index]

        expanded_attr = expand_attr(correspond_attr, class_name, dataset_path)

        # 对各个类别的属性向量进行堆叠
        if attr_count == 1:
            split_attrs = expanded_attr
        else:
            split_attrs = np.vstack((split_attrs, expanded_attr))

        attr_count += 1

    writer(split_name, 'attributes', split_attrs)