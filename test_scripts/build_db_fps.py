from src.candidate_db import CandidateDatabase
import time

t0 = time.time()
db = CandidateDatabase()
db.load(device="cpu")
print(f"Database fully built and validated in {time.time()-t0:.1f}s")

# Test query for a test neutral mass ~ 374.0636
indices, df_cand = db.query_by_mass(374.0636, ppm_tol=10.0)
print(f"Sample query: found {len(indices)} candidates for mass 374.0636 at 10 ppm")
print(df_cand.head(3))
