"""Restore a permanently persisted Protouch Initial model bundle from Supabase.
Verifies chunk count, byte size and SHA-256 before writing the joblib file.
"""
import os,base64,hashlib,sys
from pathlib import Path
from supabase import create_client

version=sys.argv[1] if len(sys.argv)>1 else "initial-0.2.0"
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
a=sb.table("protouch_initial_artifacts").select("*").eq("model_version",version).single().execute().data
chunks=sb.table("protouch_initial_artifact_chunks").select("chunk_no,payload_base64").eq("artifact_id",a["id"]).order("chunk_no").execute().data
if len(chunks)!=int(a["total_chunks"]):
    raise RuntimeError(f"Incomplete artifact: expected {a['total_chunks']} chunks, found {len(chunks)}")
blob=b"".join(base64.b64decode(x["payload_base64"]) for x in chunks)
if len(blob)!=int(a["size_bytes"]):
    raise RuntimeError(f"Size mismatch: expected {a['size_bytes']}, got {len(blob)}")
sha=hashlib.sha256(blob).hexdigest()
if sha!=a["checksum_sha256"]:
    raise RuntimeError(f"Checksum mismatch: expected {a['checksum_sha256']}, got {sha}")
Path("artifacts").mkdir(exist_ok=True)
out=Path("artifacts")/a["artifact_name"]
out.write_bytes(blob)
print({"model_version":version,"artifact_id":a["id"],"path":str(out),"bytes":len(blob),"sha256":sha,"verified":True})
