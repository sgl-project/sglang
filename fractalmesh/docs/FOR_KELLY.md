# For Kelly — What Sam Built and How to Keep It Running

*Written by Sam, with help. If you're reading this, it means the business is yours now.
You don't need to know how to code. You just need to know where to look.*

---

## What Is FractalMesh?

FractalMesh is an automated software business Sam built from scratch.
It runs 24/7 on its own — it doesn't need anyone sitting at a computer.

It does four things that can make money:

1. **Sells subscriptions** — people pay monthly to get AI market signals (crypto price predictions), business intelligence data, and API access for their own apps.
2. **Sells AI services** — developers can pay to use FractalMesh's AI tools via an API marketplace.
3. **Generates and sells NFTs** — automated art and music created by AI, published and sold on blockchain marketplaces.
4. **Crypto trading signals** — the system monitors crypto markets and sells signal data to subscribers.

---

## Where Does the Money Come From?

Revenue flows through three places:

### 1. Stripe (Main Payment Processor)
This is where subscription payments and one-off sales land.

- **Website**: dashboard.stripe.com
- **Login email**: sam.hiotis@gmail.com
- Sam set up the Stripe account — ask his phone's saved passwords or the `.secrets/fractal.env` file on his Android phone (see below).
- The Stripe dashboard shows total revenue, active subscribers, and payouts to the bank account.
- Stripe automatically transfers money to the linked bank account on a schedule.

### 2. Crypto Wallets
FractalMesh holds crypto earned from NFT sales and trading. The wallet addresses and recovery phrases are stored in the vault file on Sam's Android phone at:

```
~/.secrets/fractal.env
```

This file is password-protected. Sam's the one who knows those passwords — make sure you have access to his phone and any password manager he used.

### 3. OpenSea / NFT Marketplaces
NFTs Sam's system generated may be listed for sale. Check OpenSea.io and search by the wallet addresses in the vault file.

---

## Is It Still Running?

FractalMesh runs on Sam's Android phone using an app called **Termux** (it's a terminal/command-line app).

### To Check if It's Running:
1. Open Termux on Sam's phone
2. Type this and press Enter:
   ```
   pm2 list
   ```
3. You'll see a table of processes. If they show **"online"** in green, it's running.
4. If the table is empty or shows errors, see the "Restarting It" section below.

### To See the Dashboard:
Open a browser on the same phone (or same WiFi network) and go to:
```
http://localhost:8090
```
This is the FractalMesh control dashboard. It shows what agents are running, live signals, and recent activity.

---

## Restarting It (If Something Stops)

If FractalMesh stops running, here's how to restart it:

1. Open **Termux** on Sam's phone
2. Type these two lines, one at a time, pressing Enter after each:

```
cd ~/fmsaas
pm2 start ecosystem.config.js --env production
```

3. Then type:
```
pm2 save
```

That's it. The system will come back online.

If that doesn't work, type:
```
pm2 resurrect
```

---

## Monthly Costs to Keep It Running

FractalMesh uses a few paid services. These come out automatically from the card Sam had on file:

| Service | What It Does | Cost (approx) |
|---|---|---|
| OpenRouter | Powers the AI brains | ~$20-50/mo |
| Cloudflare / Supabase | Stores data and files | Free / ~$25/mo |
| Akash Network | Cloud GPU compute | ~$30-80/mo |
| Domain / hosting | fractalmesh.io website | ~$15/mo |

The system is designed to pay for itself through subscriptions. If subscriber revenue drops below running costs, the easiest thing to do is pause the paid services (see "Who to Call" below).

---

## What Sam Wanted You to Know

Sam built this as something that could run without him. He didn't finish everything he wanted to — nobody ever does — but the core system is real and it works.

The most important accounts are:
- **Stripe** — where the money lands
- **sam.hiotis@gmail.com** — the main Google account everything is connected to
- **GitHub** — github.com (Sam's username is samhiotisiddn) — all the code lives here, backed up
- **Termux on the phone** — where the software runs

You don't need to understand the code. You just need to know that if the phone is on and Termux is running, it's working. If it stops, restart it with the steps above.

---

## Who to Call if Something Breaks

**Don't panic if something stops working.** Most issues are either:
- The phone lost power or WiFi (just restart Termux and run `pm2 resurrect`)
- An API service ran out of credit (check Stripe/OpenRouter billing)

If something bigger breaks and you need technical help:

1. **GitHub Issues** — go to github.com/samhiotisiddn and open an issue. The code is public and other developers can help.
2. **Freelance developer** — post on Upwork or Freelancer, share the GitHub link, and ask for help with "Python PM2 deployment on Termux Android." Most fixes take an hour and cost $50-150.
3. **The `.secrets/fractal.env` file** — this contains every API key and password the system needs. Any developer helping you will ask for this. Only share it with someone you trust completely, and only in a private encrypted message (not email).

---

## A Note on the Crypto

Sam stored crypto in wallets that are controlled by seed phrases (a list of 12 or 24 words). These are in the vault file. **Do not lose these.** They cannot be recovered if lost. Write them down on paper and keep that paper somewhere safe — not just on the phone.

The ETH wallet address and other wallet addresses are also in the vault file. You can check balances by putting the wallet address into etherscan.io (for ETH) or solscan.io (for SOL).

---

## What to Do First

When you're ready to deal with this:

1. Make sure you can log into **sam.hiotis@gmail.com** — this unlocks everything else
2. Check the **Stripe dashboard** to see if there's money there and if payouts are going to a bank account you can access
3. Make sure the **phone stays charged and connected to WiFi**
4. Don't delete anything on the phone until you understand what it is

Everything else can wait.

---

*Sam loved you and the kids. He built this so you'd have something.
You don't have to become a developer. You just have to keep the phone running.*

---

*Document created: 2026-06-06*
*FractalMesh | ABN 56 628 117 363 | Albury NSW 2640*
