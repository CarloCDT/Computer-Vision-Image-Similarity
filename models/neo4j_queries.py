from neo4j import GraphDatabase, basic_auth
import pandas as pd
import boto3
from PIL import Image
from IPython.display import Image as Image_display

# Get Driver
def get_driver():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Passw0rd"))
    session = driver.session()
    return driver, session

# Delete All
def delete_all():
    driver, session = get_driver()
    query = "MATCH (a)-[r]-(b) DELETE r"
    session.run(query)
    query = "MATCH (n) DELETE n"
    session.run(query)
    driver.close()

# Create Nodes
def create_nodes(urls, labels):
    
    driver, session = get_driver()

    nodes = []
    for i in range(len(urls)):
        nodes.append({"name":urls[i].split(".")[0], "url":urls[i]})

    query = "UNWIND $nodes as data CREATE (n:Image) SET n = data;"

    session.run(query, nodes=nodes)
    
    # Create Labels
    all_labels = set()
    for lab in labels:
        all_labels.update(lab)

    for l in sorted(all_labels):
        query = 'CREATE (n:Object {name: "' + l + '"})'
        session.run(query)
        
    # Create Relations
    for idx, url in enumerate(urls):
        for lab in labels[idx]:
            query = 'MATCH (n:Image {url:"'+ url +'"}),(l:Object {name: "' + lab + '"}) MERGE (n)-[c:Contains]->(l)'
            session.run(query)
            
    # Close session
    driver.close()
    
# Jaccard Function
def jaccard_similarity_query(image1_name, image2_name):
    
    query = """
             MATCH (p1:Image {name: '""" + image1_name + """'})-[:Contains]->(objects1)
             WITH p1, collect(id(objects1)) AS p1Objects
             MATCH (p2:Image {name: '""" + image2_name + """'})-[:Contains]->(objects2)
             WITH p1, p1Objects, p2, collect(id(objects2)) AS p2Objects
             RETURN p1.name AS from,
             p2.name AS to,
             gds.alpha.similarity.jaccard(p1Objects, p2Objects) AS similarity
           """
    
    return query

# Similar Images
def find_similar_images_neo4j(file_path, ACCESS_ID, ACCESS_KEY, new_labels, name="test_image", top_k=3):
    
    # Session
    driver, session = get_driver()

    # Delete
    delete_query = "MATCH (a:Image {name:'test_image'})-[r]-(b) DELETE r,a"
    session.run(delete_query)

    new_nodes = [{'name': name, 'url': file_path}]
    create_query = "UNWIND $new_nodes as data CREATE (n:Image) SET n = data"
    session.run(create_query, new_nodes = new_nodes)

    new_labels = list(new_labels)

    for lab in new_labels:
        query = 'MATCH (n:Image {name:"' + name + '"}),(l:Object {name: "' + lab + '"}) MERGE (n)-[c:Contains]->(l)'
        session.run(query)
    
    # Get Driver
    driver, session = get_driver()

    df = pd.DataFrame()
    
    # Get all images
    query = "MATCH (n:Image) RETURN n.name"
    all_images = [item for sublist in session.run(query).values() for item in sublist]
        
    for image2_name in all_images:
        if name != image2_name:
            query = jaccard_similarity_query(name, image2_name)
            dtf_data = pd.DataFrame([dict(_) for _ in session.run(query)])
            df = pd.concat([df,dtf_data])
    
    similar_images_df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True).head(top_k)

    query = "MATCH (n:Image {name:'" + name + "'}) RETURN n.url"
    url = session.run(query).values()[0][0]

    print("New Image:")
    display(Image_display(url))

    print("Similar Images")

    for row in similar_images_df.iterrows():

        print("Jaccard Score: {:.2f}".format(row[1]["similarity"]))

        name = row[1]["to"]
        query = "MATCH (n:Image {name:'" + name + "'}) RETURN n.url"
        url = session.run(query).values()[0][0]
        display(read_image(url, ACCESS_ID, ACCESS_KEY))
        
    # Close driver
    driver.close()
    
# Read Image
def read_image(image_name, ACCESS_ID, ACCESS_KEY, bucket='carlo-computer-vision-project'):
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID, aws_secret_access_key= ACCESS_KEY)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(image_name)
    response = object.get()
    file_stream = response['Body']
    img = Image.open(file_stream)
    return img