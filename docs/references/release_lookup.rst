Release Lookup
==============

Find which SGLang release first included a specific PR or commit.

.. raw:: html

   <style>
       .release-lookup-container {
           background-color: #ffffff;
           padding: 2rem;
           border-radius: 12px;
           box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
           max-width: 600px;
           margin: 1.5rem 0;
       }

       .release-lookup-container .input-group {
           display: flex;
           gap: 10px;
           margin-bottom: 1.2rem;
       }

       .release-lookup-container input[type="text"] {
           flex: 1;
           padding: 10px 14px;
           border: 2px solid #e2e8f0;
           border-radius: 8px;
           font-size: 0.95rem;
           outline: none;
           transition: border-color 0.2s;
           color: #1e293b;
       }

       .release-lookup-container input[type="text"]:focus {
           border-color: #3b82f6;
           box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
       }

       .release-lookup-container input[type="text"]::placeholder {
           color: #94a3b8;
       }

       .release-lookup-container .rl-btn {
           padding: 10px 20px;
           background-color: #3b82f6;
           color: white;
           border: none;
           border-radius: 8px;
           font-size: 0.95rem;
           font-weight: 600;
           cursor: pointer;
           transition: background-color 0.2s;
       }

       .release-lookup-container .rl-btn:hover {
           background-color: #2563eb;
       }

       .release-lookup-container .rl-btn:disabled {
           background-color: #cbd5e1;
           cursor: not-allowed;
       }

       .release-lookup-container .rl-result {
           margin-top: 1rem;
           text-align: left;
           display: none;
       }

       .release-lookup-container .rl-result.visible {
           display: block;
       }

       .release-lookup-container .rl-result-content {
           padding: 1rem;
           border-radius: 8px;
           margin-bottom: 0.75rem;
       }

       .release-lookup-container .rl-success {
           background-color: #f0fdf4;
           border: 1px solid #bbf7d0;
           color: #166534;
       }

       .release-lookup-container .rl-error {
           background-color: #fef2f2;
           border: 1px solid #fecaca;
           color: #991b1b;
       }

       .release-lookup-container .rl-row {
           display: flex;
           justify-content: space-between;
           margin-bottom: 0.4rem;
           align-items: baseline;
       }

       .release-lookup-container .rl-row:last-child {
           margin-bottom: 0;
       }

       .release-lookup-container .rl-label {
           font-weight: 600;
           margin-right: 1rem;
           min-width: 70px;
       }

       .release-lookup-container .rl-tag-link {
           color: #3b82f6;
           text-decoration: none;
           font-weight: bold;
           font-size: 1.05rem;
       }

       .release-lookup-container .rl-tag-link:hover {
           text-decoration: underline;
       }

       .release-lookup-container .rl-badge {
           display: inline-block;
           padding: 2px 8px;
           border-radius: 12px;
           font-size: 0.75rem;
           font-weight: 600;
           text-transform: uppercase;
       }

       .release-lookup-container .rl-badge-main {
           background-color: #dbeafe;
           color: #1e40af;
       }

       .release-lookup-container .rl-badge-gateway {
           background-color: #f3e8ff;
           color: #6b21a8;
       }

       .release-lookup-container .rl-status {
           margin-top: 0.8rem;
           font-size: 0.85rem;
           color: #64748b;
           min-height: 18px;
       }

       .release-lookup-container .rl-loader {
           display: inline-block;
           width: 16px;
           height: 16px;
           border: 3px solid rgba(59, 130, 246, 0.2);
           border-radius: 50%;
           border-top-color: #3b82f6;
           animation: rl-spin 1s linear infinite;
           margin-right: 6px;
           vertical-align: text-bottom;
       }

       @keyframes rl-spin {
           to { transform: rotate(360deg); }
       }
   </style>

   <div class="release-lookup-container">
       <div class="input-group">
           <input type="text" id="rlQueryInput" placeholder="PR # (e.g. 1425), PR URL, or commit hash" autocomplete="off" />
           <button class="rl-btn" id="rlSearchBtn" disabled>Search</button>
       </div>
       <div id="rlLoading" style="display:none; color:#64748b; margin-bottom:0.8rem;">
           <span class="rl-loader"></span> Loading index…
       </div>
       <div class="rl-result" id="rlResult"></div>
       <div class="rl-status" id="rlStatus">Initializing…</div>
   </div>

   <script>
   (function() {
       var INDEX_URL = '/release_lookup/release_index.json';
       var SHORT_HASH_LEN = 8;
       var tagIndex = null, tagsArray = null, sortedCommitKeys = null;

       var input = document.getElementById('rlQueryInput');
       var btn = document.getElementById('rlSearchBtn');
       var resultDiv = document.getElementById('rlResult');
       var loadingDiv = document.getElementById('rlLoading');
       var statusDiv = document.getElementById('rlStatus');

       function formatDate(iso) {
           if (!iso) return 'Unknown';
           try { return new Date(iso).toLocaleDateString('en-US', {year:'numeric',month:'long',day:'numeric'}); }
           catch(e) { return iso; }
       }

       function getTagInfo(ref) {
           var tag = tagsArray[ref];
           return { name: tag[0], date: tag[1], type: tag[2] === 1 ? 'gateway' : 'main' };
       }

       function parseTagRef(ref) {
           if (typeof ref === 'string' && /^[mg]\d+$/.test(ref))
               return { type: ref[0], idx: parseInt(ref.slice(1)) };
           return null;
       }

       function prefixSearch(prefix) {
           if (!sortedCommitKeys) return null;
           var lo = 0, hi = sortedCommitKeys.length;
           while (lo < hi) {
               var mid = (lo + hi) >>> 1;
               if (sortedCommitKeys[mid] < prefix) lo = mid + 1; else hi = mid;
           }
           if (lo < sortedCommitKeys.length && sortedCommitKeys[lo].indexOf(prefix) === 0)
               return sortedCommitKeys[lo];
           return null;
       }

       function loadIndex() {
           loadingDiv.style.display = 'block';
           statusDiv.textContent = 'Downloading index…';
           fetch(INDEX_URL)
               .then(function(r) {
                   if (!r.ok) throw new Error('Index not found. It is generated on each release.');
                   return r.json();
               })
               .then(function(data) {
                   tagsArray = data.t;
                   tagIndex = { prs: data.p, commits: data.c };
                   sortedCommitKeys = Object.keys(tagIndex.commits).sort();
                   var tagCount = tagsArray.length;
                   var prCount = Object.keys(tagIndex.prs).length;
                   statusDiv.textContent = 'Ready. Indexed ' + tagCount + ' releases and ' + prCount + ' PRs.';
                   btn.disabled = false;
               })
               .catch(function(e) {
                   statusDiv.innerHTML = '<span style="color:#991b1b;">Error: ' + e.message + '</span>';
                   btn.disabled = true;
               })
               .finally(function() { loadingDiv.style.display = 'none'; });
       }

       function search() {
           if (!tagIndex) return;
           var raw = input.value.trim();
           if (!raw) return;
           resultDiv.style.display = 'none';
           resultDiv.classList.remove('visible');
           resultDiv.innerHTML = '';

           var queryType = 'unknown', key = raw;
           var urlMatch = raw.match(/\/pull\/(\d+)/);
           if (urlMatch) { key = urlMatch[1]; queryType = 'pr'; }
           else if (/^#?\d+$/.test(raw)) { key = raw.replace('#',''); queryType = 'pr'; }
           else if (/^[0-9a-fA-F]{7,40}$/.test(raw)) { key = raw.toLowerCase(); queryType = 'commit'; }

           var tagData = null;
           if (queryType === 'pr') {
               tagData = tagIndex.prs[key];
           } else if (queryType === 'commit') {
               var sk = key.slice(0, SHORT_HASH_LEN);
               tagData = tagIndex.commits[sk];
               if (!tagData) { var mk = prefixSearch(sk); if (mk) tagData = tagIndex.commits[mk]; }
           }

           renderResult(tagData, queryType, key);
       }

       function renderResult(tagData, queryType, key) {
           resultDiv.innerHTML = '';
           resultDiv.style.display = 'block';
           void resultDiv.offsetWidth;
           resultDiv.classList.add('visible');

           var tagRefs = [];
           if (tagData) {
               if (typeof tagData === 'string') {
                   var p = parseTagRef(tagData);
                   if (p) tagRefs.push(p.idx);
               } else if (typeof tagData === 'object') {
                   if ('m' in tagData) tagRefs.push(tagData.m);
                   if ('g' in tagData) tagRefs.push(tagData.g);
               }
           }

           if (tagRefs.length === 0) {
               var label = queryType === 'pr' ? 'PR #' + key : 'Commit ' + key.substring(0,7);
               var c = document.createElement('div');
               c.className = 'rl-result-content rl-error';
               c.innerHTML = '<div class="rl-row"><span class="rl-label">Status</span><span>Not Found</span></div>';
               var msg = document.createElement('div');
               msg.style.marginTop = '6px';
               var s = document.createElement('strong');
               s.textContent = label;
               msg.appendChild(document.createTextNode('The ' + queryType + ' '));
               msg.appendChild(s);
               msg.appendChild(document.createTextNode(' has not been included in any release yet, or is not in the index.'));
               c.appendChild(msg);
               resultDiv.appendChild(c);
               return;
           }

           var repoUrl = 'https://github.com/sgl-project/sglang';
           for (var i = 0; i < tagRefs.length; i++) {
               var info = getTagInfo(tagRefs[i]);
               var tagUrl = repoUrl + '/releases/tag/' + encodeURIComponent(info.name);
               var badgeClass = info.type === 'gateway' ? 'rl-badge-gateway' : 'rl-badge-main';
               var box = document.createElement('div');
               box.className = 'rl-result-content rl-success';
               box.innerHTML =
                   '<div class="rl-row"><span class="rl-label">Release</span><a target="_blank" class="rl-tag-link"></a></div>' +
                   '<div class="rl-row"><span class="rl-label">Date</span><span class="rl-date"></span></div>' +
                   '<div class="rl-row"><span class="rl-label">Module</span><span class="rl-badge ' + badgeClass + ' rl-module"></span></div>';
               var link = box.querySelector('.rl-tag-link');
               link.href = tagUrl;
               link.textContent = info.name;
               box.querySelector('.rl-date').textContent = formatDate(info.date);
               box.querySelector('.rl-module').textContent = info.type;
               resultDiv.appendChild(box);
           }
       }

       btn.addEventListener('click', search);
       input.addEventListener('keypress', function(e) { if (e.key === 'Enter') search(); });
       loadIndex();
   })();
   </script>
