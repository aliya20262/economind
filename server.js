import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';
import Anthropic from '@anthropic-ai/sdk';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { z } from 'zod';

const app = express();
const PORT = process.env.PORT || 3001;

// ── Clients ──────────────────────────────────────────────────────
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── Middleware ───────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL, credentials: true }));
app.use(express.json({ limit: '1mb' }));

// Rate limiting
const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 20, message: 'Too many requests' });
const apiLimiter  = rateLimit({ windowMs: 60 * 1000, max: 30 });
const agentLimiter = rateLimit({ windowMs: 60 * 1000, max: 5, message: 'Agent analysis limit reached' });

// ── Auth middleware ──────────────────────────────────────────────
function requireAuth(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
}

async function requirePro(req, res, next) {
  const { data: user } = await supabase
    .from('users').select('plan').eq('id', req.user.id).single();
  if (!user || user.plan !== 'pro')
    return res.status(403).json({ error: 'Pro plan required', upgrade: true });
  next();
}

// ── Validation schemas ───────────────────────────────────────────
const BusinessSchema = z.object({
  industry:    z.string(),
  stage:       z.string(),
  revenue:     z.number().min(0),
  customers:   z.number().min(1),
  avgPrice:    z.number().min(0),
  fixedCosts:  z.number().min(0),
  varCost:     z.number().min(0),
  marketing:   z.number().min(0),
  monthlyBurn: z.number().min(0),
  capital:     z.number().min(0),
  growth:      z.number().min(0).max(300),
  s1: z.number().min(0).max(100).optional(),
  s2: z.number().min(0).max(100).optional(),
  s3: z.number().min(0).max(100).optional(),
});

// ════════════════════════════════════════════════════════════════
// AUTH ROUTES
// ════════════════════════════════════════════════════════════════

// Register
app.post('/api/auth/register', authLimiter, async (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name)
    return res.status(400).json({ error: 'Missing fields' });
  if (password.length < 8)
    return res.status(400).json({ error: 'Password too short' });

  const hash = await bcrypt.hash(password, 12);

  const { data, error } = await supabase.from('users').insert({
    email: email.toLowerCase(),
    password_hash: hash,
    name,
    plan: 'free',
    analyses_count: 0,
  }).select('id, email, name, plan').single();

  if (error) {
    if (error.code === '23505') return res.status(409).json({ error: 'Email already exists' });
    return res.status(500).json({ error: 'Registration failed' });
  }

  const token = jwt.sign({ id: data.id, email: data.email }, process.env.JWT_SECRET, { expiresIn: '7d' });
  res.json({ token, user: { id: data.id, email: data.email, name: data.name, plan: data.plan } });
});

// Login
app.post('/api/auth/login', authLimiter, async (req, res) => {
  const { email, password } = req.body;
  const { data: user } = await supabase
    .from('users').select('*').eq('email', email.toLowerCase()).single();

  if (!user || !(await bcrypt.compare(password, user.password_hash)))
    return res.status(401).json({ error: 'Invalid credentials' });

  const token = jwt.sign({ id: user.id, email: user.email }, process.env.JWT_SECRET, { expiresIn: '7d' });
  res.json({
    token,
    user: { id: user.id, email: user.email, name: user.name, plan: user.plan, analyses_count: user.analyses_count }
  });
});

// Me
app.get('/api/auth/me', requireAuth, async (req, res) => {
  const { data } = await supabase
    .from('users').select('id, email, name, plan, analyses_count, created_at').eq('id', req.user.id).single();
  res.json(data);
});

// ════════════════════════════════════════════════════════════════
// ANALYSIS ROUTES
// ════════════════════════════════════════════════════════════════

// Free plan: basic analysis (3 analyses/day limit)
app.post('/api/analyse/basic', requireAuth, apiLimiter, async (req, res) => {
  // Check daily limit for free users
  const { data: user } = await supabase.from('users').select('plan, analyses_today').eq('id', req.user.id).single();
  if (user.plan === 'free' && (user.analyses_today || 0) >= 3)
    return res.status(429).json({ error: 'Daily limit reached (3/day on free plan)', upgrade: true });

  const parsed = BusinessSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: 'Invalid data', details: parsed.error.flatten() });

  const biz = parsed.data;
  const calc = computeMetrics(biz);

  const prompt = buildAnalysisPrompt(biz, calc, 'basic');
  let aiResp = {};
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    });
    const clean = msg.content[0].text.replace(/```json|```/g, '').trim();
    aiResp = JSON.parse(clean);
  } catch {
    aiResp = buildFallback(biz, calc);
  }

  // Save analysis
  const { data: analysis } = await supabase.from('analyses').insert({
    user_id: req.user.id,
    business_data: biz,
    metrics: calc,
    ai_response: aiResp,
    tier: 'basic',
  }).select('id').single();

  // Increment counters
  await supabase.from('users').update({
    analyses_count: supabase.rpc('increment', { row_id: req.user.id }),
    analyses_today: (user.analyses_today || 0) + 1,
  }).eq('id', req.user.id);

  res.json({ analysisId: analysis.id, metrics: calc, ai: aiResp });
});

// Pro plan: full analysis with econometrics
app.post('/api/analyse/pro', requireAuth, requirePro, apiLimiter, async (req, res) => {
  const parsed = BusinessSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: 'Invalid data' });

  const biz = parsed.data;
  const calc = computeMetrics(biz);
  const econo = computeEconometrics(biz, calc);

  const prompt = buildAnalysisPrompt(biz, calc, 'pro');
  let aiResp = {};
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1500,
      messages: [{ role: 'user', content: prompt }],
    });
    aiResp = JSON.parse(msg.content[0].text.replace(/```json|```/g, '').trim());
  } catch {
    aiResp = buildFallback(biz, calc);
  }

  const { data: analysis } = await supabase.from('analyses').insert({
    user_id: req.user.id,
    business_data: biz,
    metrics: calc,
    econometrics: econo,
    ai_response: aiResp,
    tier: 'pro',
  }).select('id').single();

  res.json({ analysisId: analysis.id, metrics: calc, econometrics: econo, ai: aiResp });
});

// Get analysis history
app.get('/api/analyses', requireAuth, async (req, res) => {
  const { data } = await supabase
    .from('analyses')
    .select('id, created_at, tier, metrics, business_data')
    .eq('user_id', req.user.id)
    .order('created_at', { ascending: false })
    .limit(20);
  res.json(data || []);
});

// Get single analysis
app.get('/api/analyses/:id', requireAuth, async (req, res) => {
  const { data } = await supabase
    .from('analyses')
    .select('*')
    .eq('id', req.params.id)
    .eq('user_id', req.user.id)
    .single();
  if (!data) return res.status(404).json({ error: 'Not found' });
  res.json(data);
});

// ════════════════════════════════════════════════════════════════
// AI AGENT COUNCIL (Pro only)
// ════════════════════════════════════════════════════════════════
app.post('/api/agents/run', requireAuth, requirePro, agentLimiter, async (req, res) => {
  const parsed = BusinessSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: 'Invalid data' });

  const biz = parsed.data;
  const calc = computeMetrics(biz);
  const ctx = buildContext(biz, calc);

  const agents = [
    {
      name: 'Strategic Analyst',
      role: 'strategist',
      prompt: `You are a strategic analyst. Analyse this ${ctx}. Focus on revenue trajectory, growth metrics, market positioning, and strategic opportunities. 3-4 sentences. End with "VERDICT:" and one clear recommendation.`
    },
    {
      name: "Devil's Advocate",
      role: 'skeptic',
      prompt: `You are a devil's advocate. Challenge everything about this ${ctx}. Find the 2-3 most serious weaknesses. 3-4 sentences. End with "VERDICT:" and risk level: LOW/MEDIUM/HIGH/CRITICAL.`
    },
    {
      name: 'Risk Officer',
      role: 'risk',
      prompt: `You are a risk officer. Analyse this ${ctx}. Focus on burn rate, runway, concentration risk, stress scenarios. 3-4 sentences. End with "VERDICT:" and safe runway in months.`
    },
    {
      name: 'Growth Advisor',
      role: 'growth',
      prompt: `You are a growth advisor. Analyse this ${ctx}. Focus on unit economics, LTV:CAC, customer efficiency, scaling prerequisites. 3-4 sentences. End with "VERDICT:" and scale-readiness score /100.`
    },
  ];

  // Run all 4 agents in parallel
  const agentResults = await Promise.allSettled(
    agents.map(async (agent) => {
      const msg = await anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 250,
        messages: [{ role: 'user', content: agent.prompt }],
      });
      return { ...agent, output: msg.content[0].text };
    })
  );

  const results = agentResults.map((r, i) => ({
    ...agents[i],
    output: r.status === 'fulfilled' ? r.value.output : 'Analysis unavailable.',
    success: r.status === 'fulfilled',
  }));

  // Verifier agent reads all 4 and synthesises
  const verifierPrompt = `You are a senior investment committee chair. Four AI agents analysed the same business. Read their findings, check for contradictions, and produce a final consensus verdict.

${results.map(r => `${r.name.toUpperCase()}:\n${r.output}`).join('\n\n')}

Business data: ${ctx}

Your task:
1. Identify the strongest points each agent made
2. Note any contradictions between agents
3. Produce a FINAL CONSENSUS including:
   - Investment score 0-100
   - 3-sentence synthesis
   - Top priority action for the founder
   - Confidence level: LOW/MEDIUM/HIGH

Format as JSON: {"score": 72, "synthesis": "...", "action": "...", "confidence": "HIGH", "contradictions": "...", "strongest_insight": "..."}`;

  let consensus = null;
  try {
    const verifierMsg = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 500,
      messages: [{ role: 'user', content: verifierPrompt }],
    });
    consensus = JSON.parse(verifierMsg.content[0].text.replace(/```json|```/g, '').trim());
  } catch {
    consensus = {
      score: calc.sustainabilityScore,
      synthesis: 'Agent council completed analysis. See individual verdicts above.',
      action: 'Review agent findings and prioritise the most critical recommendation.',
      confidence: 'MEDIUM',
      contradictions: 'Unable to synthesise automatically.',
      strongest_insight: results[0]?.output || '',
    };
  }

  // Save to DB
  await supabase.from('agent_sessions').insert({
    user_id: req.user.id,
    business_data: biz,
    agent_results: results,
    consensus,
  });

  res.json({ agents: results, consensus });
});

// ════════════════════════════════════════════════════════════════
// AI CHAT (Pro: unlimited, Free: 10/day)
// ════════════════════════════════════════════════════════════════
app.post('/api/chat', requireAuth, apiLimiter, async (req, res) => {
  const { message, analysisId } = req.body;
  if (!message) return res.status(400).json({ error: 'Message required' });

  // Get analysis context
  let ctx = '';
  if (analysisId) {
    const { data } = await supabase.from('analyses').select('*').eq('id', analysisId).eq('user_id', req.user.id).single();
    if (data) ctx = buildContext(data.business_data, data.metrics);
  }

  const prompt = ctx
    ? `You are EconoMind's AI assistant. The user has analysed their business:\n${ctx}\n\nAnswer their question concisely and specifically. 2-4 sentences max.\n\nQuestion: ${message}`
    : `You are EconoMind's AI assistant for business financial analysis. Answer: ${message}`;

  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 300,
      messages: [{ role: 'user', content: prompt }],
    });
    res.json({ reply: msg.content[0].text });
  } catch {
    res.status(500).json({ error: 'AI unavailable' });
  }
});

// ════════════════════════════════════════════════════════════════
// STRIPE PAYMENTS
// ════════════════════════════════════════════════════════════════

// Create checkout session
app.post('/api/billing/checkout', requireAuth, async (req, res) => {
  const { data: user } = await supabase.from('users').select('email, stripe_customer_id').eq('id', req.user.id).single();

  let customerId = user.stripe_customer_id;
  if (!customerId) {
    const customer = await stripe.customers.create({ email: user.email, metadata: { userId: req.user.id } });
    customerId = customer.id;
    await supabase.from('users').update({ stripe_customer_id: customerId }).eq('id', req.user.id);
  }

  const session = await stripe.checkout.sessions.create({
    customer: customerId,
    mode: 'subscription',
    payment_method_types: ['card'],
    line_items: [{ price: process.env.STRIPE_PRO_PRICE_ID, quantity: 1 }],
    success_url: `${process.env.FRONTEND_URL}/dashboard?upgraded=true`,
    cancel_url: `${process.env.FRONTEND_URL}/pricing`,
    metadata: { userId: req.user.id },
  });

  res.json({ url: session.url });
});

// Create billing portal session
app.post('/api/billing/portal', requireAuth, async (req, res) => {
  const { data: user } = await supabase.from('users').select('stripe_customer_id').eq('id', req.user.id).single();
  if (!user?.stripe_customer_id) return res.status(400).json({ error: 'No billing account' });

  const session = await stripe.billingPortal.sessions.create({
    customer: user.stripe_customer_id,
    return_url: `${process.env.FRONTEND_URL}/dashboard`,
  });
  res.json({ url: session.url });
});

// Stripe webhook — keeps DB in sync
app.post('/api/billing/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  let event;
  try {
    event = stripe.webhooks.constructEvent(req.body, req.headers['stripe-signature'], process.env.STRIPE_WEBHOOK_SECRET);
  } catch {
    return res.status(400).send('Webhook error');
  }

  const customerId = event.data.object.customer;
  if (!customerId) return res.json({ received: true });

  const { data: user } = await supabase.from('users').select('id').eq('stripe_customer_id', customerId).single();
  if (!user) return res.json({ received: true });

  if (event.type === 'customer.subscription.created' || event.type === 'customer.subscription.updated') {
    const sub = event.data.object;
    const plan = sub.status === 'active' ? 'pro' : 'free';
    await supabase.from('users').update({ plan, stripe_subscription_id: sub.id }).eq('id', user.id);
  }

  if (event.type === 'customer.subscription.deleted') {
    await supabase.from('users').update({ plan: 'free', stripe_subscription_id: null }).eq('id', user.id);
  }

  res.json({ received: true });
});

// ════════════════════════════════════════════════════════════════
// CORE CALCULATION ENGINE (server-side, protected)
// ════════════════════════════════════════════════════════════════
function computeMetrics(d) {
  const cm    = d.avgPrice - d.varCost;
  const cmPct = d.avgPrice > 0 ? (cm / d.avgPrice) * 100 : 0;
  const tvc   = d.varCost * d.customers;
  const tc    = d.fixedCosts + tvc + d.marketing;
  const np    = d.revenue - tc;
  const gm    = d.revenue > 0 ? ((d.revenue - tvc) / d.revenue) * 100 : 0;
  const om    = d.revenue > 0 ? (np / d.revenue) * 100 : 0;
  const fcb   = tc > 0 ? (d.fixedCosts / tc) * 100 : 0;
  const beu   = cm > 0 ? Math.ceil(d.fixedCosts / cm) : 0;
  const ber   = beu * d.avgPrice;
  const mbe   = (d.growth > 0 && beu > d.customers) ? Math.ceil((beu - d.customers) / (d.customers * d.growth / 100)) : 0;
  const rpc   = d.customers > 0 ? d.revenue / d.customers : 0;
  const ltv   = (d.avgPrice * 12) * (cmPct / 100);
  const cac   = d.avgPrice * 0.4;
  const ltvCac = cac > 0 ? ltv / cac : 0;
  const burn  = np < 0 ? Math.abs(np) : d.monthlyBurn;
  const runway = burn > 0 ? d.capital / burn : 999;
  const oli   = (d.revenue - d.fixedCosts) !== 0 ? d.revenue / (d.revenue - d.fixedCosts) : 0;

  let cr, cl;
  if (d.s1 !== undefined) {
    const mx = Math.max(d.s1||0, d.s2||0, d.s3||0);
    cr = mx>=70?'HIGH':mx>=50?'MEDIUM':'LOW';
    cl = mx>=70?'red':mx>=50?'orange':'green';
  } else { cr='N/A'; cl='gray'; }

  const grs = Math.min(100, Math.max(0, Math.round(
    (gm>60?20:gm>30?12:6)+(ltvCac>3?20:ltvCac>1?12:5)+(runway>12?20:runway>6?12:4)+(om>15?20:om>0?12:0)+(cr==='LOW'?20:cr==='MEDIUM'?10:0)
  )));
  const ss = Math.min(100, Math.max(0, Math.round(
    (gm*0.25)+(ltvCac*5)+(om*0.5)+(runway>6?20:runway*3)+(cr==='LOW'?5:cr==='MEDIUM'?2:0)
  )));

  return { np, gm, om, fcb, cm, cmPct, tvc, tc, beu, ber, mbe, rpc, ltv, cac, ltvCac, burn, runway, oli, cr, cl, grs, ss };
}

function computeEconometrics(d, c) {
  const rev  = Array.from({length:12},(_,i)=>d.revenue*Math.pow(1+d.growth/100,i));
  const prof = rev.map(r=>r-c.tc);
  // Simple OLS: Rev ~ t
  const t=[1,2,3,4,5,6,7,8,9,10,11,12];
  const mx=6.5, mr=rev.reduce((a,v)=>a+v,0)/12;
  const Sxx=t.reduce((s,v)=>s+(v-mx)**2,0);
  const Sxy=t.reduce((s,v,i)=>s+(v-mx)*(rev[i]-mr),0);
  const b=Sxy/Sxx, a=mr-b*mx;
  const res=rev.map((v,i)=>v-(a+b*t[i]));
  const SSres=res.reduce((s,r)=>s+r**2,0);
  const SStot=rev.reduce((s,v)=>s+(v-mr)**2,0);
  const r2=SStot>0?1-SSres/SStot:1;
  const lerner=c.cmPct/100;
  const epsilon=lerner>0?-1/lerner:-1;
  return { olsRevTrend:{a,b,r2}, elasticity:epsilon, lerner:lerner*100 };
}

function buildContext(biz, calc) {
  return `${biz.stage}-stage ${biz.industry}: Revenue $${biz.revenue.toLocaleString()}/mo, ${biz.customers} customers, avg price $${biz.avgPrice}, fixed costs $${biz.fixedCosts.toLocaleString()}, var cost $${biz.varCost}/unit, marketing $${biz.marketing.toLocaleString()}/mo. Net profit $${Math.round(calc.np).toLocaleString()}, gross margin ${calc.gm.toFixed(1)}%, LTV:CAC ${calc.ltvCac.toFixed(1)}:1, runway ${calc.runway>99?'infinite':calc.runway.toFixed(1)+' months'}, sustainability ${calc.ss}/100.`;
}

function buildAnalysisPrompt(biz, calc, tier) {
  const base = `You are a senior startup financial analyst. ${buildContext(biz, calc)}\n\nReturn ONLY JSON (no markdown):`;
  const keys = tier === 'pro'
    ? `{"rec1":"...","rec2":"...","rec3":"...","profitAnalysis":"...","breakEvenAnalysis":"...","marginAnalysis":"...","unitEconAnalysis":"...","costAnalysis":"...","leverageAnalysis":"...","cashAnalysis":"...","riskAnalysis":"...","scenarioAnalysis":"...","priceAnalysis":"...","growthAnalysis":"...","masterAnalysis":"...","growthReadinessScore":${calc.grs},"voltageRating":${calc.ltvCac.toFixed(2)}}`
    : `{"rec1":"...","rec2":"...","rec3":"...","profitAnalysis":"...","masterAnalysis":"...","growthReadinessScore":${calc.grs}}`;
  return `${base}\n${keys}`;
}

function buildFallback(biz, calc) {
  return {
    rec1: `Reduce fixed cost burden from ${calc.fcb.toFixed(0)}% — highest leverage on profitability.`,
    rec2: `A 5-10% price increase on $${biz.avgPrice} would disproportionately boost margins.`,
    rec3: `Optimise marketing spend of $${biz.marketing.toLocaleString()}/mo toward highest-converting channels.`,
    profitAnalysis: `Operating margin ${calc.om.toFixed(1)}% with ${calc.fcb.toFixed(0)}% fixed cost burden.`,
    masterAnalysis: `Sustainability ${calc.ss}/100. Priority: optimise costs, improve margins, diversify revenue.`,
    growthReadinessScore: calc.grs,
  };
}

// ── Health check ─────────────────────────────────────────────────
app.get('/api/health', (_, res) => res.json({ status: 'ok', version: '4.0.0' }));

app.listen(PORT, () => console.log(`EconoMind API running on :${PORT}`));
